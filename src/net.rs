// SPDX-License-Identifier: MIT
//
// Original work for this crate.
//
// One security decision, in one place, because two components now make it.
//
// The progress server binds loopback and refuses anything else without an
// explicit opt-in. A distribution coordinator has the *opposite* requirement —
// it is useless unless other nodes can reach it — but it is the same decision
// and must stay the same conscious choice, so the policy lives here rather than
// being restated (and eventually diverging) in each server.
//
// A compute node is a shared machine. Binding `0.0.0.0` there publishes a
// user's run — the shape of their data, their op chain, their file paths — to
// everyone else on that network, silently, with no authentication, and the user
// finds out never. So a non-loopback bind is refused unless it was asked for by
// name.
//
// There is no authentication in either server. That is a consequence of the
// loopback default rather than an oversight, and it is also why the public bind
// is gated rather than merely discouraged: on a cluster the honest answer is a
// private network plus a deliberate flag, not a token that would have to be
// distributed by the same rendezvous that has not authenticated anybody either.

use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

use crate::error::{Error, Result};

/// The bind policy, separated so it can be tested without opening a socket.
///
/// Loopback covers `127.0.0.0/8` and `::1`. Everything else — including a
/// specific interface address on a compute node, which is the shape this
/// mistake usually takes — needs the flag.
pub fn check_bind(addr: &SocketAddr, allow_public: bool) -> Result<()> {
    if addr.ip().is_loopback() || allow_public {
        return Ok(());
    }
    Err(Error::invalid(format!(
        "refusing to bind {addr}: it is not a loopback address, so anyone who can \
         reach this machine on the network could watch this run, and there is no \
         authentication. The usual answer is to keep the default bind and forward \
         the port over SSH:\n    \
         ssh -N -L {port}:127.0.0.1:{port} <user>@<node>\n\
         If a public bind really is what you want, ask for it by name: \
         `--allow-public` on the command line, `Options::allow_public` in code.",
        port = addr.port()
    )))
}

/// What to publish as the address other nodes should connect to.
///
/// **Not** simply the bound address, and this is the caveat the distribution
/// design says to design for rather than discover. A cluster commonly has a
/// management network and a fabric network with different names for the same
/// host (`node001` and `node001-ib`), and the hostname a scheduler hands you
/// resolves to one of them, not necessarily the one with the bandwidth or even
/// the one the other nodes can route to. Binding `0.0.0.0` makes that worse,
/// not better: the bound address is then `0.0.0.0`, which is not an address
/// anybody can connect to.
///
/// So the advertised address is a **separate, configurable** quantity:
///
/// * `advertise` given — use it, resolved to a socket address. This is the flag
///   an operator reaches for when the default picked the wrong interface.
/// * otherwise, if the bound IP is unspecified (`0.0.0.0` / `[::]`), fall back
///   to the local hostname, because the wildcard cannot be published.
/// * otherwise the bound address is already specific, and is what to publish.
pub fn advertised_addr(bound: SocketAddr, advertise: Option<&str>) -> Result<SocketAddr> {
    if let Some(text) = advertise {
        return resolve_one(text, bound.port());
    }
    if !bound.ip().is_unspecified() {
        return Ok(bound);
    }
    let host = hostname().ok_or_else(|| {
        Error::invalid(format!(
            "bound {bound}, which is a wildcard address and cannot be published to \
             other nodes, and this machine's hostname could not be read. Pass the \
             address other nodes should use with `--advertise HOST[:PORT]`."
        ))
    })?;
    resolve_one(&host, bound.port()).map_err(|err| {
        Error::invalid(format!(
            "{err}\nThis came from falling back to the hostname {host:?} because the \
             bind address {bound} is a wildcard. Clusters often have several \
             interfaces with different names; name the right one with \
             `--advertise HOST[:PORT]`."
        ))
    })
}

/// `HOST`, `HOST:PORT` or `IP:PORT`, resolved to exactly one socket address.
///
/// Takes the first resolution rather than trying them all: a coordinator
/// publishes one address, and a host with several is precisely the case the
/// `--advertise` flag exists to disambiguate by hand.
pub fn resolve_one(text: &str, default_port: u16) -> Result<SocketAddr> {
    let with_port = if text.parse::<SocketAddr>().is_ok() {
        text.to_string()
    } else if text.matches(':').count() == 1
        && text
            .rsplit(':')
            .next()
            .is_some_and(|tail| !tail.is_empty() && tail.bytes().all(|byte| byte.is_ascii_digit()))
    {
        // `host:port`, where the host is a name rather than an IPv6 literal.
        text.to_string()
    } else {
        format!("{text}:{default_port}")
    };
    with_port
        .to_socket_addrs()
        .map_err(|err| {
            Error::invalid(format!("cannot resolve {with_port:?} to an address: {err}"))
        })?
        .next()
        .ok_or_else(|| Error::invalid(format!("{with_port:?} resolved to no address at all")))
}

/// This machine's hostname, or `None` where it cannot be read.
///
/// Read from the environment first — a scheduler that knows which interface it
/// wants sets it — and from `/etc/hostname` or `uname` otherwise. Deliberately
/// no dependency for this: it is a fallback for a value that should have been
/// passed in.
fn hostname() -> Option<String> {
    for name in ["BLOCKFLOW_ADVERTISE_HOST", "HOSTNAME"] {
        if let Ok(value) = std::env::var(name) {
            if !value.trim().is_empty() {
                return Some(value.trim().to_string());
            }
        }
    }
    if let Ok(text) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// A textual form that survives a round trip through a rendezvous.
pub fn addr_to_string(addr: &SocketAddr) -> String {
    match addr.ip() {
        IpAddr::V4(ip) => format!("{ip}:{}", addr.port()),
        IpAddr::V6(ip) => format!("[{ip}]:{}", addr.port()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_loopback_bind_is_refused_unless_asked_for_by_name() {
        let public: SocketAddr = "0.0.0.0:8731".parse().unwrap();
        let error = check_bind(&public, false).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("not a loopback address"), "{text}");
        assert!(text.contains("ssh -N -L"), "{text}");
        check_bind(&public, true).unwrap();
        for local in ["127.0.0.1:1", "127.0.0.5:80", "[::1]:8731"] {
            check_bind(&local.parse().unwrap(), false).unwrap();
        }
        assert!(check_bind(&"10.0.0.4:8731".parse().unwrap(), false).is_err());
    }

    #[test]
    fn a_specific_bind_is_published_as_itself() {
        let bound: SocketAddr = "10.0.0.4:9100".parse().unwrap();
        assert_eq!(advertised_addr(bound, None).unwrap(), bound);
    }

    #[test]
    fn an_explicit_advertisement_wins_over_the_bind() {
        let bound: SocketAddr = "0.0.0.0:9100".parse().unwrap();
        // The fabric-interface case: same host, different name, and only the
        // operator knows which one the other nodes should use.
        assert_eq!(
            advertised_addr(bound, Some("127.0.0.9")).unwrap(),
            "127.0.0.9:9100".parse::<SocketAddr>().unwrap()
        );
        assert_eq!(
            advertised_addr(bound, Some("127.0.0.9:7777")).unwrap(),
            "127.0.0.9:7777".parse::<SocketAddr>().unwrap()
        );
    }

    #[test]
    fn a_wildcard_bind_is_never_published_as_the_wildcard() {
        // Whatever it falls back to, `0.0.0.0` is not an address a peer can
        // connect to and must never be what gets written to a rendezvous.
        let bound: SocketAddr = "0.0.0.0:9100".parse().unwrap();
        if let Ok(advertised) = advertised_addr(bound, None) {
            assert!(!advertised.ip().is_unspecified(), "{advertised}");
        }
    }

    #[test]
    fn addresses_round_trip_through_text() {
        for text in ["127.0.0.1:9000", "[::1]:9000"] {
            let addr: SocketAddr = text.parse().unwrap();
            assert_eq!(addr_to_string(&addr).parse::<SocketAddr>().unwrap(), addr);
        }
    }
}
