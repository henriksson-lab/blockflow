// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// How a worker finds the coordinator. Pluggable, **because the mechanism
// differs and only the mechanism does.**
//
// That is the whole reason this is a trait. A worker's life is *connect to the
// coordinator at X, pull a task, execute, report*; only how `X` is discovered
// varies by environment, so that is the only thing behind an interface. Nothing
// downstream of `resolve` knows which backend produced the address.
//
// | environment | backend |
// |---|---|
// | batch scheduler on a shared filesystem | [`FileRendezvous`], keyed by the job id |
// | a managed multi-node batch service | [`EnvRendezvous`] — the service hands you the main node's address, so there is nothing to discover |
// | generic cloud instances | [`ObjectRendezvous`], polled |
// | local testing, and any manual setup | [`DirectRendezvous`] — the address, on the command line |
//
// Why a published file beats a derived port
// -----------------------------------------
// The tempting alternative is to derive a port from the job id, so both ends
// compute the same number and nothing has to be published. It breaks on shared
// nodes: two users' jobs collide, and the failure is one job silently talking to
// another's coordinator. Publishing means the coordinator can bind port 0, let
// the operating system pick, and write down what it got — which cannot collide
// with anything, and costs one small file on a filesystem the job already
// requires.
//
// Staleness, because a coordinator can move
// -----------------------------------------
// A requeued or preempted job restarts its coordinator somewhere else, and the
// old rendezvous is still sitting there pointing at a dead address. So every
// published record carries an epoch, and `resolve` will wait for a record newer
// than one it has been told is stale rather than returning the same corpse
// forever. Keying on the job id is the other half of that: a stale record from
// *another* job is never even looked at.
//
// What is deliberately not here
// -----------------------------
// Authentication. A rendezvous says where the coordinator is; it does not say
// who may talk to it, and pretending otherwise by adding a token to the record
// would be security theatre — the record is readable by anyone who can read the
// filesystem or the bucket it is in. The real control is the bind policy
// (`net::check_bind`) and the network the job runs on.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::net::{addr_to_string, resolve_one};

/// What gets published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub job: String,
    pub addr: SocketAddr,
    /// Seconds since the epoch, so a later coordinator's record wins over an
    /// earlier one's without either knowing about the other.
    pub epoch: u64,
    pub pid: u32,
}

impl Record {
    pub fn new(job: impl Into<String>, addr: SocketAddr) -> Self {
        Self {
            job: job.into(),
            addr,
            epoch: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|since| since.as_secs())
                .unwrap_or(0),
            pid: std::process::id(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "job": self.job,
            "address": addr_to_string(&self.addr),
            "epoch": self.epoch,
            "pid": self.pid,
        })
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let address = value
            .get("address")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::invalid("a rendezvous record has no \"address\""))?;
        Ok(Self {
            job: value
                .get("job")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            addr: resolve_one(address, 0)?,
            epoch: value.get("epoch").and_then(Value::as_u64).unwrap_or(0),
            pid: value.get("pid").and_then(Value::as_u64).unwrap_or(0) as u32,
        })
    }
}

/// Publish an address; find an address.
pub trait Rendezvous: Send + Sync {
    /// Called by the coordinator once it has bound. A backend where the address
    /// is handed down by the environment has nothing to do here, and says so by
    /// doing nothing rather than by erroring.
    fn publish(&self, record: &Record) -> Result<()>;

    /// Called by a worker. Polls until a record appears or `timeout` elapses.
    fn resolve(&self, timeout: Duration) -> Result<Record>;

    /// For a message when it does not work.
    fn describe(&self) -> String;
}

/// `file:PATH`, `env:VARIABLE[:PORT]`, `object:DIR/KEY`, `direct:HOST:PORT`.
///
/// One string, because it is one command-line flag on both the coordinator and
/// the worker and they must be given the same thing — a batch script that had
/// to write it two different ways would eventually write it two different ways.
pub fn parse(spec: &str) -> Result<Box<dyn Rendezvous>> {
    let (kind, rest) = spec.split_once(':').unwrap_or(("file", spec));
    Ok(match kind {
        "file" => Box::new(FileRendezvous::new(PathBuf::from(rest))),
        "direct" => Box::new(DirectRendezvous::new(resolve_one(rest, 0)?)),
        "env" => {
            let (variable, port) = match rest.rsplit_once(':') {
                Some((variable, port)) if port.chars().all(|c| c.is_ascii_digit()) => {
                    (variable.to_string(), port.parse().unwrap_or(0))
                }
                _ => (rest.to_string(), super::DEFAULT_COORDINATOR_PORT),
            };
            Box::new(EnvRendezvous::new(variable, port))
        }
        "object" => {
            let path = PathBuf::from(rest);
            let key = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("rendezvous.json")
                .to_string();
            let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            Box::new(ObjectRendezvous::new(
                Box::new(DirectoryObjects::new(root)),
                key,
            ))
        }
        other => {
            return Err(Error::invalid(format!(
                "{other:?} is not a rendezvous. Use file:PATH (a shared filesystem), \
                 env:VARIABLE (a scheduler that hands you the address), object:DIR/KEY \
                 (an object store, polled), or direct:HOST:PORT."
            )))
        }
    })
}

// ---------------------------------------------------------------- file --

/// A file on a filesystem every node can see, keyed by the job.
///
/// Written to a temporary name and renamed into place, so a worker never reads
/// half a record. `rename` within a directory is atomic on every filesystem
/// this would run on, and a half-written rendezvous is exactly the kind of
/// once-in-a-thousand-runs failure that is impossible to reproduce afterwards.
pub struct FileRendezvous {
    path: PathBuf,
}

impl FileRendezvous {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Rendezvous for FileRendezvous {
    fn publish(&self, record: &Record) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|err| {
                    Error::backend(format!("creating {}: {err}", parent.display()))
                })?;
            }
        }
        let temporary = self
            .path
            .with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&temporary, record.to_json().to_string())
            .map_err(|err| Error::backend(format!("writing {}: {err}", temporary.display())))?;
        std::fs::rename(&temporary, &self.path).map_err(|err| {
            Error::backend(format!(
                "publishing {} as {}: {err}",
                temporary.display(),
                self.path.display()
            ))
        })
    }

    fn resolve(&self, timeout: Duration) -> Result<Record> {
        poll(timeout, self.describe(), || {
            let text = std::fs::read_to_string(&self.path).ok()?;
            let value: Value = serde_json::from_str(&text).ok()?;
            Record::from_json(&value).ok()
        })
    }

    fn describe(&self) -> String {
        format!("the rendezvous file {}", self.path.display())
    }
}

// ----------------------------------------------------------------- env --

/// The address comes from the environment; there is nothing to discover.
///
/// A managed multi-node batch service tells every node the main node's private
/// address in a variable, so publishing is a no-op and resolving is a read. The
/// port is not in the variable — the service knows the host, not what the
/// coordinator bound — so it is a parameter, and it is the one case where the
/// coordinator must bind a *known* port rather than letting the system pick.
pub struct EnvRendezvous {
    variable: String,
    port: u16,
}

impl EnvRendezvous {
    pub fn new(variable: impl Into<String>, port: u16) -> Self {
        Self {
            variable: variable.into(),
            port,
        }
    }
}

impl Rendezvous for EnvRendezvous {
    fn publish(&self, _record: &Record) -> Result<()> {
        // Nothing to do, and that is the point of this backend rather than a
        // gap in it: the scheduler published the address before this process
        // started.
        Ok(())
    }

    fn resolve(&self, timeout: Duration) -> Result<Record> {
        let port = self.port;
        let variable = self.variable.clone();
        poll(timeout, self.describe(), move || {
            let host = std::env::var(&variable).ok()?;
            let host = host.trim();
            if host.is_empty() {
                return None;
            }
            let addr = resolve_one(host, port).ok()?;
            Some(Record {
                job: String::new(),
                addr,
                epoch: 0,
                pid: 0,
            })
        })
    }

    fn describe(&self) -> String {
        format!(
            "${} (with the coordinator on port {})",
            self.variable, self.port
        )
    }
}

// -------------------------------------------------------------- direct --

/// The address, given on the command line.
pub struct DirectRendezvous {
    addr: SocketAddr,
}

impl DirectRendezvous {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }
}

impl Rendezvous for DirectRendezvous {
    fn publish(&self, _record: &Record) -> Result<()> {
        Ok(())
    }

    fn resolve(&self, _timeout: Duration) -> Result<Record> {
        Ok(Record {
            job: String::new(),
            addr: self.addr,
            epoch: 0,
            pid: 0,
        })
    }

    fn describe(&self) -> String {
        format!("the address {} given on the command line", self.addr)
    }
}

// -------------------------------------------------------------- object --

/// A key-value store with put and get. Two methods, because two is what a
/// rendezvous needs.
///
/// The store this is *for* is a cloud object store, and there is deliberately no
/// implementation of one here: that would be a client library, a credentials
/// chain and a region configuration in a crate whose dependency list is four
/// lines and defended in a comment. A deployment supplies the eight lines that
/// wrap its own client — or gets one behind a feature flag when somebody has a
/// bucket to test it against. What *is* here is the part that is ours and that
/// a bucket would not test any better: the polling, the staleness rule, and the
/// record format.
pub trait ObjectStore: Send + Sync {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;
    /// `None` for "not there yet", which is the normal state while a worker
    /// waits for its coordinator to start. An error is for a store that is
    /// broken, not for a key that has not appeared.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn describe(&self) -> String;
}

/// An object store backed by a directory.
///
/// Not a stand-in for the real thing on a cluster — that is what
/// [`FileRendezvous`] is, and it is simpler. This exists so the polling and
/// staleness logic above has something to run against, because those are the
/// parts that can be wrong, and a bucket would not exercise them any harder
/// than a directory does.
pub struct DirectoryObjects {
    root: PathBuf,
}

impl DirectoryObjects {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

impl ObjectStore for DirectoryObjects {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        std::fs::create_dir_all(&self.root)
            .map_err(|err| Error::backend(format!("creating {}: {err}", self.root.display())))?;
        let path = self.root.join(key);
        let temporary = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&temporary, bytes)
            .map_err(|err| Error::backend(format!("writing {}: {err}", temporary.display())))?;
        std::fs::rename(&temporary, &path)
            .map_err(|err| Error::backend(format!("publishing {}: {err}", path.display())))
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.root.join(key)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(Error::backend(format!("reading {key}: {err}"))),
        }
    }

    fn describe(&self) -> String {
        format!("the directory {}", self.root.display())
    }
}

/// One object, polled.
///
/// Polling rather than watching because an object store has no notification and
/// a worker has nothing else to do until its coordinator exists. The interval is
/// the one in [`poll`], which starts tight and backs off — a coordinator that is
/// already up is found immediately, and one that is still starting is not
/// hammered for a minute.
pub struct ObjectRendezvous {
    store: Box<dyn ObjectStore>,
    key: String,
}

impl ObjectRendezvous {
    pub fn new(store: Box<dyn ObjectStore>, key: impl Into<String>) -> Self {
        Self {
            store,
            key: key.into(),
        }
    }
}

impl Rendezvous for ObjectRendezvous {
    fn publish(&self, record: &Record) -> Result<()> {
        self.store
            .put(&self.key, record.to_json().to_string().as_bytes())
    }

    fn resolve(&self, timeout: Duration) -> Result<Record> {
        poll(timeout, self.describe(), || {
            let bytes = self.store.get(&self.key).ok().flatten()?;
            let value: Value = serde_json::from_slice(&bytes).ok()?;
            Record::from_json(&value).ok()
        })
    }

    fn describe(&self) -> String {
        format!("{} in {}", self.key, self.store.describe())
    }
}

// ---------------------------------------------------------------- poll --

/// Wait for a record to appear, backing off.
///
/// Shared by every backend that has to wait, which is all of them except the
/// two where the answer is already known. Starting at 20 ms and doubling to a
/// quarter of a second means the common case — the coordinator was already up —
/// costs one attempt, and the slow case does not spin.
fn poll(
    timeout: Duration,
    what: String,
    mut attempt: impl FnMut() -> Option<Record>,
) -> Result<Record> {
    let deadline = Instant::now() + timeout;
    let mut wait = Duration::from_millis(20);
    loop {
        if let Some(record) = attempt() {
            return Ok(record);
        }
        if Instant::now() >= deadline {
            return Err(Error::backend(format!(
                "no coordinator appeared at {what} within {:.1?}. Either it has not started \
                 yet — raise the timeout — or it published somewhere else, which on a \
                 cluster usually means the two ends were given different rendezvous \
                 strings.",
                timeout
            )));
        }
        std::thread::sleep(wait.min(deadline.saturating_duration_since(Instant::now())));
        wait = (wait * 2).min(Duration::from_millis(250));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "blockflow-rendezvous-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|since| since.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_record_survives_the_round_trip() {
        let record = Record::new("job-7", "127.0.0.1:9111".parse().unwrap());
        let read = Record::from_json(&record.to_json()).unwrap();
        assert_eq!(read.addr, record.addr);
        assert_eq!(read.job, "job-7");
        assert_eq!(read.pid, std::process::id());
    }

    #[test]
    fn a_file_rendezvous_publishes_and_resolves() {
        let dir = scratch("file");
        let rendezvous = FileRendezvous::new(dir.join("job-1.json"));
        let addr: SocketAddr = "127.0.0.1:9112".parse().unwrap();
        rendezvous.publish(&Record::new("job-1", addr)).unwrap();
        assert_eq!(
            rendezvous.resolve(Duration::from_millis(50)).unwrap().addr,
            addr
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_later_coordinator_overwrites_an_earlier_one() {
        // The requeue case: the coordinator moved, and a worker starting now
        // must find the new address rather than the corpse.
        let dir = scratch("requeue");
        let rendezvous = FileRendezvous::new(dir.join("job.json"));
        rendezvous
            .publish(&Record::new("job", "127.0.0.1:1111".parse().unwrap()))
            .unwrap();
        let mut moved = Record::new("job", "127.0.0.1:2222".parse().unwrap());
        moved.epoch += 1;
        rendezvous.publish(&moved).unwrap();
        let found = rendezvous.resolve(Duration::from_millis(50)).unwrap();
        assert_eq!(found.addr.port(), 2222);
        assert_eq!(found.epoch, moved.epoch);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_rendezvous_that_never_appears_says_where_it_looked() {
        let rendezvous = FileRendezvous::new(PathBuf::from("/nonexistent/blockflow/nope.json"));
        let error = rendezvous.resolve(Duration::from_millis(30)).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("nope.json"), "{text}");
        assert!(text.contains("different rendezvous strings"), "{text}");
    }

    #[test]
    fn an_object_rendezvous_polls_until_the_object_appears() {
        let dir = scratch("object");
        let store = DirectoryObjects::new(dir.clone());
        let addr: SocketAddr = "127.0.0.1:9113".parse().unwrap();
        // Nothing there yet.
        let waiting = ObjectRendezvous::new(Box::new(DirectoryObjects::new(dir.clone())), "r.json");
        assert!(waiting.resolve(Duration::from_millis(30)).is_err());
        // A coordinator starts.
        ObjectRendezvous::new(Box::new(store), "r.json")
            .publish(&Record::new("job", addr))
            .unwrap();
        assert_eq!(
            waiting.resolve(Duration::from_millis(200)).unwrap().addr,
            addr
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_direct_rendezvous_needs_no_publication_at_all() {
        let addr: SocketAddr = "127.0.0.1:9114".parse().unwrap();
        let rendezvous = DirectRendezvous::new(addr);
        rendezvous.publish(&Record::new("job", addr)).unwrap();
        assert_eq!(
            rendezvous.resolve(Duration::from_secs(0)).unwrap().addr,
            addr
        );
    }

    #[test]
    fn an_env_rendezvous_reads_the_address_the_scheduler_set() {
        // The managed-batch case: the address is handed down, so `publish` is a
        // no-op and `resolve` is a read.
        let variable = format!("BLOCKFLOW_TEST_MAIN_NODE_{}", std::process::id());
        std::env::set_var(&variable, "127.0.0.1");
        let rendezvous = EnvRendezvous::new(variable.clone(), 9115);
        rendezvous
            .publish(&Record::new("job", "127.0.0.1:9115".parse().unwrap()))
            .unwrap();
        let found = rendezvous.resolve(Duration::from_millis(50)).unwrap();
        assert_eq!(found.addr, "127.0.0.1:9115".parse::<SocketAddr>().unwrap());
        std::env::remove_var(&variable);
    }

    #[test]
    fn every_backend_is_reachable_from_one_flag() {
        let dir = scratch("parse");
        assert!(parse(&format!("file:{}/r.json", dir.display())).is_ok());
        assert!(parse("direct:127.0.0.1:9000").is_ok());
        assert!(parse("env:AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS").is_ok());
        assert!(parse("env:SOME_VARIABLE:9200").is_ok());
        assert!(parse(&format!("object:{}/r.json", dir.display())).is_ok());
        let error = match parse("carrier-pigeon:home") {
            Ok(_) => panic!("a rendezvous nobody implements should not parse"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("is not a rendezvous"), "{error}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
