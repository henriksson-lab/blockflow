use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Scratch(PathBuf);

pub type ScratchDir = Scratch;

impl Scratch {
    pub fn new(prefix: &str, name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let unique = NEXT.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "blockflow-{prefix}-{}-{name}-{unique}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        Self(path)
    }

    pub fn path(&self) -> &PathBuf {
        &self.0
    }

    pub fn join(&self, tail: &str) -> PathBuf {
        self.0.join(tail)
    }

    pub fn keep(self) -> PathBuf {
        let path = self.0.clone();
        std::mem::forget(self);
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
