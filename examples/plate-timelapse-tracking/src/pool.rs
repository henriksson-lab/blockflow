// SPDX-License-Identifier: MIT

//! A bounded worker pool: `jobs` threads pull items in order. Bounds total
//! concurrency — and so memory and inference load — across a whole plate
//! instead of starting every frame at once.

use std::collections::VecDeque;
use std::sync::Mutex;

use blockflow::{Error, Result};

/// Run `work` over `items` on at most `jobs` threads. Every item runs even if
/// another fails; the first error (in item order) is returned.
pub fn parallel<T, F>(items: Vec<T>, jobs: usize, work: F) -> Result<()>
where
    T: Send,
    F: Fn(T) -> Result<()> + Sync,
{
    let results = run(items, jobs, |item| work(item));
    match results.into_iter().find_map(|r| r.err()) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Run `work` over `items` on at most `jobs` threads and return every result
/// in item order.
pub fn run<T, R, F>(items: Vec<T>, jobs: usize, work: F) -> Vec<Result<R>>
where
    T: Send,
    R: Send,
    F: Fn(T) -> Result<R> + Sync,
{
    let count = items.len();
    let queue = Mutex::new(items.into_iter().enumerate().collect::<VecDeque<_>>());
    let results: Mutex<Vec<Option<Result<R>>>> = Mutex::new((0..count).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1).min(count.max(1)) {
            scope.spawn(|| loop {
                let next = queue.lock().unwrap_or_else(|p| p.into_inner()).pop_front();
                let Some((index, item)) = next else { break };
                let outcome = work(item);
                results.lock().unwrap_or_else(|p| p.into_inner())[index] = Some(outcome);
            });
        }
    });
    results
        .into_inner()
        .unwrap_or_else(|p| p.into_inner())
        .into_iter()
        .map(|r| r.unwrap_or_else(|| Err(Error::invalid("worker did not run an item"))))
        .collect()
}
