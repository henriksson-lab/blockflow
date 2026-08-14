// SPDX-License-Identifier: MIT
//
// Original work for this crate. Generalised from `crate::points`, which was
// written from the shape of the problem — a set of positions produced piecewise
// and read by extent — and is now the four-column case of this.
//
// **A row store keyed by position, with typed columns.**
//
// The problem this removes
// ------------------------
// `crate::points` holds a position and *one* `f64`. That is enough to say "there
// is something here, and it is this big", and it is not enough for anything that
// wants a row per object: a size, a total, a shape measure, three numbers about
// one thing at one place. Until this module there were exactly two channels for
// that, and both are worse than they look:
//
// * **more point sets.** One store per column, keyed by the same positions,
//   joined by the consumer. The join is on a coordinate triple that each store
//   sorted independently, so it is correct only while every store holds exactly
//   the same positions — a precondition nothing states and nothing checks;
// * **an opaque fragment blob.** `BlockOutput::fragment` takes bytes, so a
//   producer can put anything in them. What that costs is that every consumer
//   writes its own parser against a layout that is documented, at best, in the
//   producer's header — and the layouts drift, because nothing can compare them.
//
// A table is neither. One row holds one object's position and every column of
// it; the schema travels **in the blob**, so a consumer that was handed the
// wrong stream is told which column disagrees rather than reading somebody
// else's numbers as its own.
//
// What carried over from `crate::points` unchanged
// ------------------------------------------------
// Everything structural, because it was right: the two states, the canonical
// order with an intrinsic tiebreak, the streaming `scan`, the two
// interchangeable indexes, and the k-way merge that holds a cross-section rather
// than the whole span. Each of those is argued for below in its own section, and
// the arguments are the ones that module made — a point set is a table with one
// payload column, so a property that had to hold for one holds for the other for
// the same reason.
//
// Two states, and why reading early is an error rather than an emptiness
// ---------------------------------------------------------------------
// A table is [`State::Accumulating`] — blocks write, nobody reads — or
// [`State::Sealed`] — nobody writes, anybody queries. There is no third.
//
// A query on an accumulating table **fails, naming the state**. It does not
// return what has arrived so far, and it does not return nothing, because both
// of those are answers: the first is a fact about which writers happened to have
// finished, which is the decomposition showing through in the worst possible
// place, and the second is indistinguishable from a region that is genuinely
// empty. This crate fills an unwritten level with NaN rather than zeros for the
// same reason — an absence must not be able to pass for a value.
//
// Making the states a *type* rather than a convention is what turns "every
// writer must have finished" from something a caller has to remember into
// something the store enforces. It is checked at run time rather than by a
// typestate — `seal` transitions one value instead of consuming it and returning
// another — for one reason, and it is a compromise rather than a preference:
// whether every writer has finished is a *run-time* fact, known to whoever drove
// the phase, and a table is reached through a shared handle by the tasks that
// write it. A typestate would make the illegal call unwriteable, which is
// strictly better where the caller owns the value, and would put the check
// nowhere at all where it does not.
//
// The read interface is one method, and it streams
// ------------------------------------------------
// [`Table::scan`] takes a region and returns an **iterator** over the rows in
// it. That is the whole of it, and the granularity of whatever index was built
// is invisible from the outside: nothing downstream can tell a flat table from a
// gridded one except by timing it. Keeping the interface at one method is not
// economy for its own sake — it is what makes that claim checkable, because
// there is exactly one thing to check.
//
// It is an iterator rather than a `Vec` because the case this store exists for
// is the one where the answer does not fit. A whole-volume query returning a
// `Vec` allocates a second copy of the entire set, which is precisely what an
// out-of-core structure must not do; a stream borrows the index and yields.
// [`Table::query`] collects, and is kept because collecting is often what a
// caller wants — but it is now a **choice visible at the call site** rather than
// the only thing on offer. What it collects is a [`Row`] per row, which is a
// cursor into the store rather than the row's values, so even the convenience
// does not copy the data; it copies the pointers to it.
//
// Half-open, like every other region in this crate: a row at `region.start` is
// in the answer and one at `region.start + region.shape` is not.
//
// The canonical order, and why the tiebreak is the rest of the row
// ----------------------------------------------------------------
// Rows come back sorted by **the coordinate triple, lexicographically, then by
// the payload columns' bits, in schema order**. Two properties depend on it and
// both are load-bearing:
//
// * **the implementations are interchangeable.** If they returned the same set
//   in different orders, "invisible downstream" would be false the first time
//   somebody wrote the answer to a file or hashed it;
// * **a consumer that folds in store order gets a function of the data.**
//   Floating-point `+` does not associate, so a sum over an [`ColumnType::F64`]
//   column takes its value from the order it is taken in. In store order that
//   value depends on the row set and on nothing else — not on the block grid,
//   not on which worker finished first, not on whether the index was rebuilt.
//
// A row's storage *is* its sort key, and that is the whole of the mechanism:
// positions and payload are `u64` words in one array, positions first, so the
// canonical order is the lexicographic order of the row's words and the
// comparison is one `slice::cmp`. There is nowhere for a second key to come
// from.
//
// Lexicographic on the coordinate triple, rather than Morton: it makes a query a
// contiguous run on the first axis, which is what the flat index binary searches
// for, and it is readable in an error message, where a Morton code is a number
// nobody can place. Morton would give a query better locality in three
// dimensions and is the right answer if that ever dominates — it needs a fixed
// bit budget per axis, which is a limit this has no need to take on yet.
//
// **The tiebreak is the row's own payload, and the store records nothing else.**
// Not the source block, and not the order of insertion — both of those are facts
// about the run rather than about the data, and either would make the answer
// move when the cut moved. [`Table::write`] takes a block index for its error
// messages and keeps no trace of it, so a tiebreak on it is **unrepresentable**
// rather than forbidden. Comparing an `F64` column's bits is not comparing
// magnitudes (negative floats come back reversed) and that is fine, because this
// is a tiebreak and not a sort by value: what is needed is a deterministic total
// order on the payload, and the payload *is* those bits. Two rows that agree on
// every word are indistinguishable by any observation, so their relative order
// cannot matter and is not fixed.
//
// Why the order is a property of the *storage* and not of the query
// -----------------------------------------------------------------
// The obvious way to guarantee a canonical order is to sort the answer on the
// way out, in one place, so no implementation can get it wrong. **Streaming
// forbids it**: sorting requires having the whole answer, which is the copy the
// stream exists not to make. So one of three things has to give, and only one of
// them is right:
//
// * *each index sorts its own output* — back to one obligation in as many places
//   as there are indexes, and per call;
// * *the stream is unordered and the caller sorts* — which hands the
//   float-accumulation-order problem back to every consumer, and that problem is
//   half the reason this module exists;
// * **each index stores its rows already in the canonical order, so yielding
//   them in it is free.** This is the one taken.
//
// The obligation does move back into the implementations, and that is worth
// being clear about — but it changes shape on the way. It is no longer "return
// sorted", which is a promise made afresh on every call and can be broken by any
// one of them; it is "store sorted", which a constructor establishes once and a
// query cannot undo. For [`Flat`] it is what the structure already was: a sorted
// array, and a region query is a range scan that is in order because the array
// is. For [`Gridded`] it means each bucket holds its rows in order and a
// multi-bucket region is answered by a **k-way merge** — a heap over one cursor
// per live bucket, `O(log k)` per row, materialising nothing.
//
// The merge's working set is a *cross-section*, not the whole span
// ---------------------------------------------------------------
// A naive merge holds a cursor for every bucket the region touches, and for a
// whole-volume query that is every bucket — which is a fraction of the row count
// and therefore still proportional to the set. The lexicographic order removes
// it: a row in a bucket at grid position `bx` on the first axis has its first
// coordinate inside that bucket's range, so **every row of grid plane `bx`
// precedes every row of plane `bx + 1`**, and planes cannot interleave. The
// merge therefore runs one plane at a time and its heap holds at most the
// buckets of one cross-section — `|second span| * |third span|`. For a
// whole-volume query that is `O(buckets^(2/3))`, and it does not grow at all
// when the set grows along the first axis. That is the property
// `points::tests::a_stream_holds_a_cross_section_and_not_the_set` asserts, and
// it asserts it about *this* merge — the point store and the table share one
// index, so the residency claim is made once and holds for both.
//
// Two indexes, and why the choice may not show
// --------------------------------------------
// | index | shape | right when |
// |---|---|---|
// | flat | one sorted array; binary search, then an in-order scan | there are few rows |
// | gridded | sorted buckets on a coarse grid; a k-way merge over the overlapping ones | there are many |
//
// [`Table::seal`] picks from the row count. The choice is *reportable* —
// [`Table::layout`] says which was built, so a slow run can be explained — and
// it is not *observable*: the same queries return the same words either way,
// which is what `both_indexes_answer_every_query_identically` asserts directly.
// That property is the whole design. If it ever fails, the abstraction is gone
// and the two are not implementations of one thing but two things with a shared
// name.
//
// The schema, and why the type set is two
// ---------------------------------------
// A column is a **name** and one of [`ColumnType::U64`] or [`ColumnType::F64`].
// The positions are not columns: every row has exactly three, they are always
// first, and no schema can omit or reorder them. That is deliberate — the
// gridded index buckets on them and the canonical order leads with them, so a
// table without them is one no index here could serve, and a schema that could
// express it would be a shape every index would have to refuse at run time
// instead of one the type forbids.
//
// **Two payload types, because two is what the producers and consumers on either
// side of this module actually have.** `ops::detect` accumulates a voxel count
// and three position sums, all `u64`, and emits an `f64` weight; `ops::voxelize`
// consumes an `f64` weight and nothing else. Every column type is a
// compatibility surface — every index, every encoder, every future consumer and
// every error message has to handle it — so the set is the smallest one that
// loses nothing:
//
// * *narrower widths* (`u32`, `f32`) are a storage saving and not a capability:
//   every value fits the wide form, and the encoder is where bytes should be
//   recovered if they ever need to be, because that is a change no index or
//   consumer can see;
// * *a boolean* is a `U64` holding 0 or 1, which orders the same way and needs
//   no new branch anywhere;
// * *a signed integer* is the one real omission. Nothing on either side produces
//   one today. Adding it is not free even though the storage is: two's-complement
//   bits do not order the way the values do, so it would either sort negatives
//   above positives — acceptable for a *tiebreak*, and confusing to read in an
//   error message — or force the order to become type-directed, which is a
//   comparison that stops being one `slice::cmp`. That is the trade whoever needs
//   it should make deliberately, so it is written down here rather than
//   pre-empted.
//
// A schema mismatch is refused **at write time, naming the column**: the blob
// carries the producer's column names and types, so a wrong stream is caught
// where the wrong bytes arrive rather than wherever the wrong numbers eventually
// look strange.
//
// Exactness, and what a float column costs
// ----------------------------------------
// **A `U64` column is exact end to end.** It is stored as its own bits, encoded
// as its own bits, compared as its own bits, and at no point on the path from a
// producer's accumulator to a consumer's read is it converted to a float. That
// matters more than it sounds: `ops::detect`'s accumulators are `u64` precisely
// because an `f64` first moment stops being exact at a volume edge of about
// 11500 voxels, which is a volume this crate is meant to be pointed at, and a
// table that could only carry the answer as an `f64` would have thrown that away
// in transport. `a_u64_column_carries_a_split_accumulation_exactly` is the test
// that says so, with a value chosen above `2^53` so that the `f64` path
// demonstrably loses it.
//
// **An `F64` column round trips exactly and does not make addition associate.**
// The bits are preserved — `to_bits` in, `from_bits` out, no decimal and no
// rounding anywhere — so a value written is the value read. What is *not*
// recovered is associativity: summing an `F64` column over a set of rows gives
// an answer that depends on the order the sum is taken in. This store fixes that
// order, and that is the whole of what it can offer: a fold in store order is
// reproducible under every decomposition, because the order is a function of the
// row set alone. It is **not** exact, and a partial fold merged across a seam is
// not the same number as the whole fold. Where that matters — anywhere a
// quantity is accumulated per block and merged, which is every fragment-and-join
// op in `ops/` — the column has to be `U64` and the arithmetic has to be
// integer. The type set makes that possible; nothing here can make it automatic.
//
// A non-finite `F64` is refused on the way in, naming the column. A NaN is not
// ordered by its own bits in the way a reader would expect, and any sum over a
// set containing one stops meaning anything at all.
//
// The layout is row-major, and what that costs
// --------------------------------------------
// One flat `Vec<u64>`, `3 + columns` words per row, positions first. A column is
// therefore strided rather than contiguous, which is the opposite of what a
// column store is for, and it is chosen because of what the interface is: the
// only read primitive is `scan`, which yields **one row at a time in a canonical
// order**, so the access pattern this has to be fast for is "the next row" — and
// under a column-major layout that is one cache line per column, per row. Row-
// major also makes a row's words contiguous, which is what lets a merge cursor
// be a slice and lets the canonical comparison be a single `slice::cmp`.
//
// What it costs is a future reader that wants one whole column and no others: it
// pays the stride. That reader does not exist yet, and when it does, **the
// layout is private** — `Row` hands out words, `scan` hands out rows, and
// neither says how they are stored — so it can change without touching a caller.
// That reversibility is why this is the right way round to be wrong.
//
// Why `Point` is a view over a table, rather than a second store
// --------------------------------------------------------------
// `crate::points` keeps `Point`, its encoding and `PointStore`'s whole public
// surface; what it no longer keeps is an implementation. A `PointStore` **is** a
// `Table` over the schema `[weight: f64]`, and a `Point` is a row of it read
// through a fixed schema.
//
// The alternative was to let the two coexist, which is safer in the small: the
// point store has a large regression net around it (`tests/voxelize.rs`,
// `tests/detect.rs`, `ops::detect`'s unit tests and its own suite), and a new
// module that touches none of them cannot break any of them. It was not taken,
// for the reason `ops/mod.rs` gives about kernels: **two implementations of one
// thing is a cost paid on every future change, not once.** Concretely, the two
// would have shared a bucket-edge derivation, a span computation, a k-way merge
// and a cross-section argument — about three hundred lines of the subtlest code
// in either module — and every fix to the merge would have had to be made twice,
// with a test to catch you if you forgot. A test that catches a forgotten second
// fix is better than nothing and worse than not having a second place to fix.
//
// What made unification cheap enough to be the obvious call, rather than a
// trade, is that a point blob and a four-word row are the **same words**:
// `encode_points` writes three coordinates and `weight.to_bits()`, which is
// exactly a row of the point schema, so the canonical order over rows is
// byte-for-byte the order over points and no existing answer moves. The point
// store's public behaviour is unchanged down to its error messages — which is
// why `Table` carries a `noun` for them — and every test in the crate passes as
// written, including `points`'s own white-box tests of the index, which now
// exercise *this* index through the point payload. That is the unification
// paying for itself immediately: the residency proof lives with one payload and
// covers both.
//
// What this deliberately does not do
// ----------------------------------
// * **No schema evolution.** A blob whose schema is not this table's is refused,
//   including one that merely has an extra column this table would ignore. A
//   reader that silently dropped a column would be reading a producer's output
//   under an assumption the producer never made, and "which columns did the run
//   actually write" would stop being answerable from the table. Widening a
//   schema is a new table.
// * **No join.** Two tables cannot be joined here, and a table cannot be joined
//   to a level. A join on the position triple is the operation that would let a
//   caller assemble a row out of several producers, and it needs a decision this
//   module has no basis for — what to do when one side has a row the other does
//   not — which is a question about the pipeline rather than about the store.
//   The row is the unit precisely so that a producer that knows the answer emits
//   it whole.
// * **It is not yet a node the planner knows about.** `docs/design/BLOCK_OPS.md`
//   §"Levels are a DAG" lists a point set as a node with a domain, a partitioning
//   and a unit; this is that node generalised, and like `PointStore` it is
//   filled by a caller after a run rather than materialised by the executor.
//   Making it a node means a lifetime, a cost model row and a place in the
//   binding plan, and each of those is a decision about `Decomposition` rather
//   than about storage.
//
// Sealing is a barrier, and the distributed sort is a later problem
// ----------------------------------------------------------------
// `seal` sorts in one place, driven by the caller: **every writer must have
// finished before it is called**, and nothing here enforces that beyond refusing
// the writes that arrive afterwards. It is not wired into the executor's phase
// machinery and it is not a distributed sort. A table larger than one node can
// hold needs a sample-and-partition sort with the buckets as the partition, and
// that is real work with its own decomposition-invariance argument — it is the
// next problem and not this one. Naming it here so that the barrier is a stated
// limit rather than a discovered one.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::fragment::{pack_u64, unpack_u64};
use crate::region::Region;

// ----------------------------------------------------------------- schema --

/// Words of a row given to the position. Three, always, and always first; see
/// the module header for why they are not columns.
pub const POSITION_WORDS: usize = 3;

/// What a payload column holds.
///
/// Two, and the module header says why that is the whole set and what a third
/// would cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    /// An exact 64-bit unsigned integer. Stored, encoded and compared as its own
    /// bits; nothing on the path converts it to a float.
    U64,
    /// A 64-bit float, carried as `to_bits` so it round trips exactly. Summing
    /// one is order-dependent — see the module header on what this store can and
    /// cannot promise about that.
    F64,
}

impl ColumnType {
    pub fn as_str(self) -> &'static str {
        match self {
            ColumnType::U64 => "u64",
            ColumnType::F64 => "f64",
        }
    }

    /// The wire tag. Explicit numbers rather than a discriminant, because the
    /// discriminant is an implementation detail that reordering the enum would
    /// change and a blob written last week would not.
    fn code(self) -> u64 {
        match self {
            ColumnType::U64 => 1,
            ColumnType::F64 => 2,
        }
    }

    fn from_code(code: u64) -> Option<Self> {
        match code {
            1 => Some(ColumnType::U64),
            2 => Some(ColumnType::F64),
            _ => None,
        }
    }
}

/// One payload column: a name and a type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    name: String,
    kind: ColumnType,
}

impl Column {
    pub fn new(name: impl Into<String>, kind: ColumnType) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }

    /// An exact integer column.
    pub fn u64(name: impl Into<String>) -> Self {
        Self::new(name, ColumnType::U64)
    }

    /// A float column. See the module header for what it costs.
    pub fn f64(name: impl Into<String>) -> Self {
        Self::new(name, ColumnType::F64)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> ColumnType {
        self.kind
    }
}

/// The payload columns of a table, in order.
///
/// The order is part of the schema and not a presentation detail: it is the
/// order the canonical tiebreak walks and the order the encoding writes, so two
/// schemas with the same columns permuted are different schemas and are refused
/// against each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    columns: Vec<Column>,
}

impl Schema {
    /// Refuses an unnamed or a repeated column.
    ///
    /// Both would make "naming the column" ambiguous, which is the one thing
    /// every diagnostic in this module rests on — a mismatch report that says
    /// `""` or names a column twice is not a report.
    pub fn new(columns: Vec<Column>) -> Result<Self> {
        for (index, column) in columns.iter().enumerate() {
            if column.name.is_empty() {
                return Err(Error::invalid(format!(
                    "column {index} of a table schema has no name. Every mismatch this module \
                     reports names the column it is about, so an unnamed column would produce a \
                     diagnostic nobody can act on."
                )));
            }
            if let Some(earlier) = columns[..index]
                .iter()
                .position(|other| other.name == column.name)
            {
                return Err(Error::invalid(format!(
                    "columns {earlier} and {index} of a table schema are both named {:?}. A \
                     repeated name makes every message about \"the column named x\" ambiguous.",
                    column.name
                )));
            }
        }
        Ok(Self { columns })
    }

    /// The schema a point set has: one `f64`, the weight.
    ///
    /// Here rather than in `crate::points` so that this module's tests can say
    /// what the special case is without depending on it, and so that the claim
    /// "a point is a row" has one definition.
    pub fn points() -> Self {
        Self {
            columns: vec![Column::f64("weight")],
        }
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// How many payload columns. The positions are not counted; see
    /// [`Self::width`].
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Words per row: the three positions plus one per payload column.
    pub fn width(&self) -> usize {
        POSITION_WORDS + self.columns.len()
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|column| column.name == name)
    }

    /// `other` is what arrived; `self` is what this table holds.
    ///
    /// Reports the **first** disagreement rather than all of them, and names the
    /// column and both types: a count mismatch is usually one insertion, and
    /// listing every column after it as "wrong" buries the one that is.
    fn check_against(&self, other: &Schema, what: &str) -> Result<()> {
        if self.columns.len() != other.columns.len() {
            return Err(Error::invalid(format!(
                "{what} declares {} column(s) — {} — and this table has {}: {}. A table's schema \
                 travels with its rows so that a stream from the wrong producer is refused here \
                 rather than read as if it were this one.",
                other.columns.len(),
                names(other),
                self.columns.len(),
                names(self)
            )));
        }
        for (index, (mine, theirs)) in self.columns.iter().zip(other.columns.iter()).enumerate() {
            if mine.name != theirs.name {
                return Err(Error::invalid(format!(
                    "column {index} of {what} is named {:?} and column {index} of this table is \
                     named {:?}. Column order is part of the schema, so two schemas holding the \
                     same names in a different order are different schemas.",
                    theirs.name, mine.name
                )));
            }
            if mine.kind != theirs.kind {
                return Err(Error::invalid(format!(
                    "column {index} of {what}, named {:?}, is {} and this table's is {}. The \
                     words would decode either way and mean something different, so the mismatch \
                     is refused here rather than carried.",
                    mine.name,
                    theirs.kind.as_str(),
                    mine.kind.as_str()
                )));
            }
        }
        Ok(())
    }
}

fn names(schema: &Schema) -> String {
    if schema.columns.is_empty() {
        return "no columns".to_string();
    }
    schema
        .columns
        .iter()
        .map(|column| format!("{}: {}", column.name, column.kind.as_str()))
        .collect::<Vec<_>>()
        .join(", ")
}

// ------------------------------------------------------------------ value --

/// One column's value, tagged with its type.
///
/// The write path speaks in these rather than in raw words so that a value of
/// the wrong type is refused at the call that supplies it, naming the column,
/// instead of becoming a `u64` that decodes as a plausible float.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    U64(u64),
    F64(f64),
}

impl Value {
    pub fn kind(self) -> ColumnType {
        match self {
            Value::U64(_) => ColumnType::U64,
            Value::F64(_) => ColumnType::F64,
        }
    }

    /// The word this value is stored as. Exact both ways: an integer is its own
    /// bits and a float is `to_bits`, so nothing on this path rounds.
    fn bits(self) -> u64 {
        match self {
            Value::U64(value) => value,
            Value::F64(value) => value.to_bits(),
        }
    }
}

/// The one place a value is checked against the column it is going into.
///
/// Shared by [`RowBuilder::push`] and [`Table::push`] so that the message is the
/// same wherever the mismatch happens.
fn checked_word(schema: &Schema, index: usize, value: Value) -> Result<u64> {
    let column = &schema.columns[index];
    if column.kind != value.kind() {
        return Err(Error::invalid(format!(
            "column {index} of this table is named {:?} and is {}, but a {} was given for it. A \
             table refuses a mismatched schema where the value is supplied, because that is the \
             last place the caller who knows what it meant is still on the stack.",
            column.name,
            column.kind.as_str(),
            value.kind().as_str()
        )));
    }
    if let Value::F64(number) = value {
        if !number.is_finite() {
            return Err(Error::invalid(format!(
                "column {index} of this table, named {:?}, was given {number}, which is not \
                 finite. The canonical order tiebreaks on the column's bits and any sum over the \
                 table would stop being a check on anything, so it is refused here rather than \
                 carried.",
                column.name
            )));
        }
    }
    Ok(value.bits())
}

// -------------------------------------------------------------------- row --

/// One row, as a view into the store.
///
/// Three words wide as a value — a schema reference and a slice — and it holds
/// no copy of anything: a `Row` is how the answer is read without the answer
/// being materialised. That is what lets [`Table::query`] collect without
/// copying the data, and it is why a row cannot outlive the table it came from.
#[derive(Clone, Copy)]
pub struct Row<'a> {
    schema: &'a Schema,
    words: &'a [u64],
}

impl<'a> Row<'a> {
    /// The position, in **volume** coordinates. Not block-local: a row travels,
    /// and a coordinate that meant something only next to the block index that
    /// wrote it would be one more thing to get wrong — and would be exactly the
    /// producer-shaped partitioning this module exists to drop.
    ///
    /// The conversion cannot truncate: every position was checked against the
    /// volume, which is `usize`, before it was stored.
    pub fn at(&self) -> [usize; 3] {
        [
            self.words[0] as usize,
            self.words[1] as usize,
            self.words[2] as usize,
        ]
    }

    pub fn schema(&self) -> &'a Schema {
        self.schema
    }

    /// The row's words: three positions, then one per column, in schema order.
    ///
    /// This is both the wire form and **the canonical sort key**, and they are
    /// the same thing on purpose — a row that compared by anything other than
    /// its own contents would have a second key to keep in step.
    pub fn words(&self) -> &'a [u64] {
        self.words
    }

    /// One column, typed.
    pub fn value(&self, column: usize) -> Result<Value> {
        let kind = self.column(column)?.kind;
        let word = self.words[POSITION_WORDS + column];
        Ok(match kind {
            ColumnType::U64 => Value::U64(word),
            ColumnType::F64 => Value::F64(f64::from_bits(word)),
        })
    }

    /// One column, as an integer. Fails naming the column if it is not one.
    pub fn u64(&self, column: usize) -> Result<u64> {
        match self.value(column)? {
            Value::U64(value) => Ok(value),
            Value::F64(_) => Err(self.wrong_type(column, ColumnType::U64)),
        }
    }

    /// One column, as a float. Fails naming the column if it is not one.
    pub fn f64(&self, column: usize) -> Result<f64> {
        match self.value(column)? {
            Value::F64(value) => Ok(value),
            Value::U64(_) => Err(self.wrong_type(column, ColumnType::F64)),
        }
    }

    /// One column by name, which is what a consumer that was handed a schema
    /// rather than an index has.
    pub fn by_name(&self, name: &str) -> Result<Value> {
        let column = self.schema.index_of(name).ok_or_else(|| {
            Error::invalid(format!(
                "this table has no column named {name:?}; it has {}",
                names(self.schema)
            ))
        })?;
        self.value(column)
    }

    fn column(&self, column: usize) -> Result<&'a Column> {
        self.schema.columns.get(column).ok_or_else(|| {
            Error::invalid(format!(
                "column {column} was asked for and this table has {}: {}",
                self.schema.columns.len(),
                names(self.schema)
            ))
        })
    }

    fn wrong_type(&self, column: usize, wanted: ColumnType) -> Error {
        let held = &self.schema.columns[column];
        Error::invalid(format!(
            "column {column} of this table is named {:?} and is {}, and was read as {}. The word \
             would decode either way and mean something different.",
            held.name,
            held.kind.as_str(),
            wanted.as_str()
        ))
    }
}

impl std::fmt::Debug for Row<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = formatter.debug_struct("Row");
        out.field("at", &self.at());
        for (index, column) in self.schema.columns.iter().enumerate() {
            match self.value(index) {
                Ok(Value::U64(value)) => out.field(&column.name, &value),
                Ok(Value::F64(value)) => out.field(&column.name, &value),
                Err(_) => out.field(&column.name, &"?"),
            };
        }
        out.finish()
    }
}

/// The canonical order, entire.
///
/// One `slice::cmp`, because the storage order of a row's words *is* the order:
/// positions first, then the payload in schema order. See the module header for
/// why that is the right strength and why the tiebreak is the payload.
fn canonical(left: &[u64], right: &[u64]) -> Ordering {
    left.cmp(right)
}

/// Half-open containment, on all three axes, read straight off a row's words.
fn holds(region: &Region, words: &[u64]) -> bool {
    (0..3).all(|axis| {
        let at = words[axis] as usize;
        at >= region.start[axis] && at - region.start[axis] < region.shape[axis]
    })
}

// ------------------------------------------------------------- row buffer --

/// Rows as one flat array of words, with the schema that says how to cut it.
///
/// Crate-visible rather than public: it is the storage, and the module header's
/// claim that the layout can change without a caller noticing depends on no
/// caller being able to name it.
pub(crate) struct RowBuffer {
    schema: Arc<Schema>,
    words: Vec<u64>,
}

impl RowBuffer {
    pub(crate) fn new(schema: Arc<Schema>) -> Self {
        Self {
            schema,
            words: Vec::new(),
        }
    }

    /// Wrap words that are already in row form.
    ///
    /// Panics on a length that is not a whole number of rows, which is an
    /// internal invariant rather than an input: every caller in this crate has
    /// already checked the length against the schema, and a partial row here
    /// would corrupt every row after it rather than fail one.
    pub(crate) fn from_words(schema: Arc<Schema>, words: Vec<u64>) -> Self {
        assert_eq!(
            words.len() % schema.width(),
            0,
            "a row buffer is a whole number of rows of {} word(s)",
            schema.width()
        );
        Self { schema, words }
    }

    fn schema(&self) -> &Schema {
        &self.schema
    }

    fn width(&self) -> usize {
        self.schema.width()
    }

    fn len(&self) -> usize {
        self.words.len() / self.width()
    }

    fn words_of(&self, row: usize) -> &[u64] {
        let width = self.width();
        &self.words[row * width..(row + 1) * width]
    }

    fn push_words(&mut self, words: &[u64]) {
        debug_assert_eq!(words.len(), self.width());
        self.words.extend_from_slice(words);
    }

    fn extend_words(&mut self, words: &[u64]) {
        debug_assert_eq!(words.len() % self.width(), 0);
        self.words.extend_from_slice(words);
    }

    fn take(&mut self) -> RowBuffer {
        RowBuffer {
            schema: Arc::clone(&self.schema),
            words: std::mem::take(&mut self.words),
        }
    }

    /// Sort the rows by `compare`, in place.
    ///
    /// Sorted as a permutation of row indices and then applied by following
    /// cycles, rather than by sorting a vector of rows or writing into a second
    /// buffer. The reason is residency and it is the same one the whole module
    /// has: `seal` is the moment the table is at its largest, and a second copy
    /// of the words taken exactly then is the allocation an out-of-core store
    /// must not make. What it costs instead is one `usize` per row for the
    /// permutation and one row of scratch — a fraction of a row's words, not a
    /// multiple of them.
    fn sort_rows(&mut self, mut compare: impl FnMut(&[u64], &[u64]) -> Ordering) {
        let width = self.width();
        let rows = self.len();
        let mut order: Vec<usize> = (0..rows).collect();
        // Unstable is safe here and stability would be meaningless anyway: the
        // comparisons this is called with are total on every pair of rows that
        // can be told apart, and two rows with identical words are
        // indistinguishable by any observation.
        order.sort_unstable_by(|&left, &right| {
            compare(
                &self.words[left * width..(left + 1) * width],
                &self.words[right * width..(right + 1) * width],
            )
        });

        // `order[destination] = source`: follow each cycle once, carrying the
        // row the cycle started on in `scratch`.
        let mut done = vec![false; rows];
        let mut scratch = vec![0u64; width];
        for start in 0..rows {
            if done[start] || order[start] == start {
                done[start] = true;
                continue;
            }
            scratch.copy_from_slice(&self.words[start * width..(start + 1) * width]);
            let mut destination = start;
            loop {
                done[destination] = true;
                let source = order[destination];
                if source == start {
                    self.words[destination * width..(destination + 1) * width]
                        .copy_from_slice(&scratch);
                    break;
                }
                // `source` has not been written yet: the cycle visits every one
                // of its members once, and each is written only as it is left.
                self.words
                    .copy_within(source * width..(source + 1) * width, destination * width);
                destination = source;
            }
        }
    }

    /// The first row index whose words fail `before`, over a buffer already
    /// sorted by it.
    fn partition_point(&self, before: impl Fn(&[u64]) -> bool) -> usize {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if before(self.words_of(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

// ------------------------------------------------------------------ index --

/// What an index has to be able to do, and nothing else.
///
/// One method, on purpose. Everything a caller could otherwise ask — how many
/// buckets, how big they are, which one a row is in — is granularity, and
/// granularity showing through is the failure this whole module is arranged
/// against. The trait is small to *forbid* variety rather than to enable it.
///
/// **The stream is in the canonical order, and an implementation meets that by
/// storing its rows in it** rather than by sorting on the way out. The module
/// header says why streaming forces this and why "store sorted" is a weaker
/// thing to ask than "return sorted": a constructor establishes it once, where
/// sorting on the way out would be a promise made afresh on every call.
///
/// **There is deliberately no way to hand a table a foreign index.** The trait
/// is public because it is the contract every guarantee in this module is
/// stated over, and a reader has to be able to see it; the two implementations
/// are private and [`Table::seal`] chooses between them. Admitting a third from
/// outside would turn "the choice is not observable" from something this module
/// tests into a claim about somebody else's code — and a third index is a change
/// to this file, where the agreement test is, rather than a plugin.
pub trait TableIndex: Send + Sync {
    /// Every row of the table that lies in `region`, in the canonical order,
    /// **without materialising them**.
    ///
    /// The returned iterator borrows the index; an implementation that collected
    /// into a `Vec` and returned its `into_iter` would satisfy the signature and
    /// defeat the point of it.
    ///
    /// **The region is copied into the stream rather than borrowed by it**, and
    /// that is the one place this differs from `crate::points`' trait as it was.
    /// A borrowed region would make a yielded [`Row`]'s lifetime the shorter of
    /// the table's and the region's, so a row could not outlive the box it was
    /// asked for — and [`Table::query`], which hands rows back to a caller whose
    /// region was a temporary, could not be written at all. It costs two
    /// three-element vectors per scan, against a scan that walks rows.
    ///
    /// `region` is rank 3 and inside the volume; [`Table::scan`] has already
    /// checked both, so an implementation may assume them.
    fn scan<'a>(&'a self, region: &Region) -> Box<dyn Iterator<Item = Row<'a>> + 'a>;
}

/// Which index a table was sealed with.
///
/// Reportable so that a run's speed can be explained; never observable in an
/// answer. See the module header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Everything in one array, sorted once. A scan is a binary search and a
    /// walk, in order because the array is.
    Flat,
    /// Sorted buckets on a coarse grid. A scan reads only the overlapping ones,
    /// merged.
    Gridded,
}

impl Layout {
    pub fn as_str(self) -> &'static str {
        match self {
            Layout::Flat => "flat",
            Layout::Gridded => "gridded",
        }
    }
}

/// At or below this many rows, [`Table::seal`] builds a [`Layout::Flat`] index.
///
/// A few thousand rows is a few tens of kilobytes in one contiguous run, which a
/// scan crosses about as fast as memory can supply it, and the bucket grid's own
/// arrays plus the per-bucket bookkeeping cost more to build and walk than the
/// scan they would save. It is a count of rows rather than of words: a wider
/// schema makes the run longer, but the per-bucket overhead the threshold is
/// weighing does not change with it.
///
/// The number is a threshold between two implementations that give the same
/// answer, so choosing it badly costs time and can never cost correctness —
/// which is exactly why it is a constant here rather than something a caller
/// sets, and why no test asserts a query result against it.
pub(crate) const FLAT_LIMIT: usize = 4_096;

/// What the bucket-edge derivation aims for: rows per bucket, on average over
/// the volume.
pub(crate) const TARGET_PER_BUCKET: usize = 64;

// ------------------------------------------------------------------- flat --

/// Everything in one array, in the canonical order.
///
/// The order is established by the constructor and nothing afterwards disturbs
/// it, which is what makes a scan in that order cost nothing.
struct Flat {
    rows: RowBuffer,
}

impl Flat {
    fn build(rows: impl Into<RowBuffer>) -> Self {
        let mut rows = rows.into();
        rows.sort_rows(canonical);
        Self { rows }
    }
}

impl TableIndex for Flat {
    fn scan<'a>(&'a self, region: &Region) -> Box<dyn Iterator<Item = Row<'a>> + 'a> {
        // The canonical order is lexicographic and the position leads it, so
        // every row of the answer lies in one contiguous run of the first axis.
        // Find its start by binary search, stop when the axis leaves the region,
        // and filter the other two as they go. Nothing here allocates per row
        // and nothing here is reordered: the run is already in the order it has
        // to be yielded in.
        let lo = region.start[0] as u64;
        let hi = (region.start[0] + region.shape[0]) as u64;
        let width = self.rows.width();
        let from = self.rows.partition_point(|words| words[0] < lo);
        let schema = self.rows.schema();
        let region = region.clone();
        Box::new(
            self.rows.words[from * width..]
                .chunks_exact(width)
                .take_while(move |words| words[0] < hi)
                .filter(move |words| holds(&region, words))
                .map(move |words| Row { schema, words }),
        )
    }
}

// ---------------------------------------------------------------- gridded --

/// Rows bucketed on a coarse grid, stored as one array plus the offsets that cut
/// it into buckets.
///
/// One allocation for the rows and one for the offsets, rather than a map of
/// vectors: the bucket count is derived to be a fraction of the row count, so
/// the offsets are small, and a bucket's rows being contiguous is what makes
/// reading one a scan rather than a chase.
///
/// **Each bucket holds its rows in the canonical order**, which is what lets a
/// scan merge them rather than gather and sort them. The whole array is sorted
/// bucket-major with the canonical order inside each bucket, so that property
/// costs one sort at construction and nothing per query.
///
/// Crate-visible, with the measurement hooks below, because `crate::points`'
/// suite drives this type directly to assert the residency bound — see the
/// module header on what the point store and the table share.
pub(crate) struct Gridded {
    edge: [usize; 3],
    counts: [usize; 3],
    /// `counts.product() + 1` offsets, in rows, into `rows`; bucket `b` is rows
    /// `starts[b]..starts[b + 1]`.
    starts: Vec<usize>,
    rows: RowBuffer,
}

/// The bucket edge, in voxels, for `rows` rows spread over `volume`.
///
/// **What it aims at.** A bucket should hold about [`TARGET_PER_BUCKET`] rows.
/// Fewer, and a query pays the per-bucket cost — a bounds computation and a
/// fresh cache line — more often than it reads anything; more, and a query for a
/// small region drags in a large multiple of what it returns, which is the whole
/// reason not to use the flat index. So the edge is the cube root of "the volume
/// one bucket's worth of rows occupies at the average density":
/// `cbrt(TARGET_PER_BUCKET * |volume| / |rows|)`.
///
/// **Cubic, in voxels.** The shape of the regions that will be asked for is not
/// known when the index is built, so no axis has earned a finer grid than
/// another. An axis shorter than the derived edge is clamped to its own length,
/// which is what makes a flat volume get a flat bucket rather than a grid one
/// bucket wide.
///
/// **Then a guard: never more buckets than rows.** The clamp above, and a volume
/// that is mostly empty, can both leave the grid with more slots than entries —
/// an index spending memory and a walk on emptiness. Doubling the edge until the
/// count comes down is crude and terminates quickly (a big enough edge is one
/// bucket), and it is a correction rather than a policy: for any set dense enough
/// to have been given this index in the first place, it does not fire.
///
/// Derived, with nothing a caller can set. A configurable bucket size would be a
/// number that changes performance and not answers, which is precisely the kind
/// of knob that gets set once, wrongly, and never revisited.
pub(crate) fn bucket_edge(volume: [usize; 3], rows: usize) -> [usize; 3] {
    let extent = volume[0] as f64 * volume[1] as f64 * volume[2] as f64;
    let wanted = (rows / TARGET_PER_BUCKET).max(1);
    let mut edge = (extent / wanted as f64).cbrt().max(1.0);
    let cap = rows.max(1);
    // Bounded by construction: each pass doubles, and an edge past the longest
    // axis leaves one bucket, which is under any cap.
    for _ in 0..64 {
        let candidate = clamp_edge(volume, edge);
        match bucket_counts(volume, candidate) {
            Some(counts)
                if counts[0]
                    .saturating_mul(counts[1])
                    .saturating_mul(counts[2])
                    <= cap =>
            {
                return candidate
            }
            _ => edge *= 2.0,
        }
    }
    volume
}

fn clamp_edge(volume: [usize; 3], edge: f64) -> [usize; 3] {
    let mut out = [1usize; 3];
    for axis in 0..3 {
        let rounded = if edge >= volume[axis] as f64 {
            volume[axis]
        } else {
            edge.ceil() as usize
        };
        out[axis] = rounded.clamp(1, volume[axis].max(1));
    }
    out
}

/// Buckets per axis, or `None` if the product would not fit a `usize`.
pub(crate) fn bucket_counts(volume: [usize; 3], edge: [usize; 3]) -> Option<[usize; 3]> {
    let mut counts = [0usize; 3];
    for axis in 0..3 {
        counts[axis] = volume[axis].div_ceil(edge[axis].max(1)).max(1);
    }
    counts[0].checked_mul(counts[1])?.checked_mul(counts[2])?;
    Some(counts)
}

impl Gridded {
    pub(crate) fn build(volume: [usize; 3], rows: impl Into<RowBuffer>) -> Self {
        let mut rows = rows.into();
        let edge = bucket_edge(volume, rows.len());
        let counts = bucket_counts(volume, edge).unwrap_or([1, 1, 1]);
        let total = counts[0] * counts[1] * counts[2];

        // Bucket-major, canonical inside a bucket. The bucket triple compares
        // lexicographically in the same order as its linear index, so one sort
        // gives both.
        rows.sort_rows(|left, right| {
            bucket_of(left, edge)
                .cmp(&bucket_of(right, edge))
                .then_with(|| canonical(left, right))
        });

        let mut starts = vec![0usize; total + 1];
        for row in 0..rows.len() {
            let linear = linear_bucket(bucket_of(rows.words_of(row), edge), counts);
            starts[linear + 1] += 1;
        }
        for index in 1..starts.len() {
            starts[index] += starts[index - 1];
        }
        Self {
            edge,
            counts,
            starts,
            rows,
        }
    }

    /// The inclusive bucket range `region` overlaps, or `None` for a region with
    /// no voxels in it.
    fn span(&self, region: &Region) -> Option<([usize; 3], [usize; 3])> {
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for axis in 0..3 {
            if region.shape[axis] == 0 {
                return None;
            }
            lo[axis] = region.start[axis] / self.edge[axis];
            hi[axis] = ((region.start[axis] + region.shape[axis] - 1) / self.edge[axis])
                .min(self.counts[axis] - 1);
            if lo[axis] > hi[axis] {
                return None;
            }
        }
        Some((lo, hi))
    }

    /// How many buckets a query for `region` would read.
    ///
    /// A measurement hook, and deliberately **not** on [`TableIndex`]: a caller
    /// that could ask this could tell a gridded table from a flat one without
    /// timing it, which is the property the module is arranged to keep. It is
    /// visible to this crate's tests, which is where the claim "a small region
    /// reads a small part of the index" is asserted as a count rather than as a
    /// duration. Compiled only for them, so that the escape hatch cannot be
    /// reached from a build that is not the one measuring it.
    #[cfg(test)]
    pub(crate) fn buckets_touched(&self, region: &Region) -> usize {
        match self.span(region) {
            None => 0,
            Some((lo, hi)) => (0..3).map(|axis| hi[axis] - lo[axis] + 1).product(),
        }
    }

    /// How many buckets the derivation produced. Test-only, for the same reason
    /// as [`Self::buckets_touched`].
    #[cfg(test)]
    pub(crate) fn buckets(&self) -> usize {
        self.counts[0] * self.counts[1] * self.counts[2]
    }

    /// The merge itself, as its own type rather than behind the `dyn`.
    ///
    /// Split out so this crate's tests can hold the concrete cursor and watch
    /// the heap, which is where the residency claim is checked. `None` is a
    /// region with no voxels in it.
    pub(crate) fn merge<'a>(&'a self, region: &Region) -> Option<GriddedScan<'a>> {
        let (lo, hi) = self.span(region)?;
        Some(GriddedScan {
            index: self,
            region: region.clone(),
            lo,
            hi,
            next_plane: lo[0],
            heap: BinaryHeap::new(),
        })
    }

    fn bucket(&self, linear: usize) -> &[u64] {
        let width = self.rows.width();
        &self.rows.words[self.starts[linear] * width..self.starts[linear + 1] * width]
    }
}

fn bucket_of(words: &[u64], edge: [usize; 3]) -> [usize; 3] {
    [
        words[0] as usize / edge[0],
        words[1] as usize / edge[1],
        words[2] as usize / edge[2],
    ]
}

fn linear_bucket(bucket: [usize; 3], counts: [usize; 3]) -> usize {
    (bucket[0] * counts[1] + bucket[1]) * counts[2] + bucket[2]
}

impl TableIndex for Gridded {
    fn scan<'a>(&'a self, region: &Region) -> Box<dyn Iterator<Item = Row<'a>> + 'a> {
        match self.merge(region) {
            None => Box::new(std::iter::empty()),
            Some(merge) => Box::new(merge),
        }
    }
}

/// One bucket's remaining rows, with the next one that qualifies already found.
///
/// Ordered **backwards** on purpose: [`BinaryHeap`] is a max-heap and the merge
/// wants the canonically smallest head, so the comparison is reversed here
/// rather than every use being wrapped in `Reverse`.
///
/// Words rather than a `Row`, because the comparison is on words and the schema
/// is the same for every head in the heap — carrying it per cursor would be a
/// pointer per bucket for a value the scan already holds.
///
/// Crate-visible only because [`GriddedScan::heap`] is, which is only because
/// `crate::points`' residency test watches the heap's length.
pub(crate) struct Head<'a> {
    row: &'a [u64],
    rest: &'a [u64],
}

impl<'a> Head<'a> {
    /// The first row of `slice` that lies in `region`, and what follows it.
    ///
    /// Rows that fail the region test are skipped here rather than filtered out
    /// in advance: filtering in advance is a copy, and a copy of a bucket is the
    /// materialisation this whole shape exists to avoid.
    fn take(mut slice: &'a [u64], width: usize, region: &Region) -> Option<Self> {
        while slice.len() >= width {
            let (row, rest) = slice.split_at(width);
            if holds(region, row) {
                return Some(Self { row, rest });
            }
            slice = rest;
        }
        None
    }
}

impl Ord for Head<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        canonical(other.row, self.row)
    }
}

impl PartialOrd for Head<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Head<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Head<'_> {}

/// A k-way merge over the buckets a region touches, one grid plane at a time.
///
/// The plane at a time is the part that matters, and the module header derives
/// it: the canonical order is lexicographic, so no row of grid plane `bx` can
/// follow a row of plane `bx + 1`, and the two therefore never need to be live
/// together. The heap holds one cursor per bucket of a single cross-section —
/// `|second span| * |third span|` — rather than one per bucket of the span, so a
/// whole-volume stream does not hold a structure proportional to the row set.
pub(crate) struct GriddedScan<'a> {
    index: &'a Gridded,
    /// Owned rather than borrowed; [`TableIndex::scan`] says why.
    region: Region,
    lo: [usize; 3],
    hi: [usize; 3],
    /// The next grid plane on the first axis to load, once the heap runs dry.
    next_plane: usize,
    pub(crate) heap: BinaryHeap<Head<'a>>,
}

impl GriddedScan<'_> {
    /// How many cursors a full cross-section needs, which is the bound the heap
    /// never exceeds.
    ///
    /// Test-only, like the other measurement hooks on [`Gridded`]: a caller who
    /// could ask this could tell a gridded table from a flat one without timing
    /// it.
    #[cfg(test)]
    pub(crate) fn width(&self) -> usize {
        (self.hi[1] - self.lo[1] + 1) * (self.hi[2] - self.lo[2] + 1)
    }
}

impl<'a> Iterator for GriddedScan<'a> {
    type Item = Row<'a>;

    fn next(&mut self) -> Option<Row<'a>> {
        let width = self.index.rows.width();
        loop {
            if let Some(head) = self.heap.pop() {
                if let Some(next) = Head::take(head.rest, width, &self.region) {
                    self.heap.push(next);
                }
                return Some(Row {
                    schema: self.index.rows.schema(),
                    words: head.row,
                });
            }
            // The plane is exhausted, so every row it held has been yielded and
            // none of the next plane's could have come before them.
            if self.next_plane > self.hi[0] {
                return None;
            }
            let first = self.next_plane;
            self.next_plane += 1;
            for second in self.lo[1]..=self.hi[1] {
                for third in self.lo[2]..=self.hi[2] {
                    let linear = linear_bucket([first, second, third], self.index.counts);
                    if let Some(head) = Head::take(self.index.bucket(linear), width, &self.region) {
                        self.heap.push(head);
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------ state --

/// Which half of its life a table is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Blocks write; nobody may read. There is no partial answer, because a
    /// partial answer is a fact about which writers happened to have finished.
    Accumulating,
    /// Nobody writes; anybody may query.
    Sealed,
}

impl State {
    pub fn as_str(self) -> &'static str {
        match self {
            State::Accumulating => "accumulating",
            State::Sealed => "sealed",
        }
    }
}

// --------------------------------------------------------------- encoding --

/// `"BFTABLE"` and a version byte, so a blob from another stream is refused by
/// its first word rather than decoded as a schema of nonsense.
const MAGIC: u64 = u64::from_be_bytes(*b"BFTABLE\0");
const VERSION: u64 = 1;

/// Rows as a blob, self-describing.
///
/// A producer builds one of these and hands the bytes to `BlockOutput::
/// fragment`; the table that receives them checks the schema in the header
/// against its own. That check is the reason the header exists: the alternative
/// is a bare array of words, which decodes under *any* schema of the right width
/// and is therefore wrong silently.
///
/// The schema costs a few tens of bytes once per blob, against rows that are
/// tens of bytes each — so the self-description is free in every case that
/// matters and expensive only for a blob with almost nothing in it, where
/// nothing matters.
pub struct RowBuilder {
    schema: Arc<Schema>,
    rows: RowBuffer,
    /// One row of working space, reused. A `Vec` per pushed row would be an
    /// allocation per object in a producer's inner loop, for a buffer that is
    /// the same size every time.
    scratch: Vec<u64>,
}

impl RowBuilder {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            rows: RowBuffer::new(Arc::clone(&schema)),
            scratch: vec![0; schema.width()],
            schema,
        }
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// One row: a position and one value per column, in schema order.
    ///
    /// Refuses a value of the wrong type or a wrong number of them, naming the
    /// column and both types. There is no position check here — the volume
    /// belongs to the table, and a builder is a producer-side object that may
    /// not have one.
    pub fn push(&mut self, at: [usize; 3], values: &[Value]) -> Result<()> {
        if values.len() != self.schema.len() {
            return Err(Error::invalid(format!(
                "a row of this table has {} value(s) — {} — and {} were given",
                self.schema.len(),
                names(&self.schema),
                values.len()
            )));
        }
        // Filled in scratch and copied in one go, so a value refused halfway
        // through leaves no half a row behind.
        for axis in 0..3 {
            self.scratch[axis] = at[axis] as u64;
        }
        for (index, value) in values.iter().enumerate() {
            self.scratch[POSITION_WORDS + index] = checked_word(&self.schema, index, *value)?;
        }
        self.rows.push_words(&self.scratch);
        Ok(())
    }

    /// How many rows have been pushed.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.len() == 0
    }

    /// The blob: a header naming every column, then the rows.
    pub fn encode(&self) -> Vec<u8> {
        let mut words = vec![MAGIC, VERSION, self.schema.len() as u64, self.len() as u64];
        for column in self.schema.columns() {
            words.push(column.kind().code());
            let bytes = column.name().as_bytes();
            words.push(bytes.len() as u64);
            for chunk in bytes.chunks(8) {
                let mut padded = [0u8; 8];
                padded[..chunk.len()].copy_from_slice(chunk);
                words.push(u64::from_le_bytes(padded));
            }
        }
        words.extend_from_slice(&self.rows.words);
        pack_u64(&words)
    }
}

/// The schema a blob was written under, without needing one to read it.
///
/// Here so that a consumer handed an unfamiliar stream can say what it is rather
/// than guess — which is the failure the opaque-blob channel had, and the one
/// this module exists partly to close.
pub fn encoded_schema(bytes: &[u8]) -> Result<Schema> {
    let words = unpack_u64(bytes)?;
    Ok(decode_header(&words)?.0)
}

/// The schema and the row count, and where the rows start.
fn decode_header(words: &[u64]) -> Result<(Schema, usize, usize)> {
    if words.len() < 4 {
        return Err(Error::invalid(format!(
            "a table blob starts with four words — a magic, a version, a column count and a row \
             count. This one is {} word(s).",
            words.len()
        )));
    }
    if words[0] != MAGIC {
        return Err(Error::invalid(
            "this blob does not start with a table's magic word, so it was written by something \
             else. Decoding it as a table would produce a schema of nonsense rather than an \
             error."
                .to_string(),
        ));
    }
    if words[1] != VERSION {
        return Err(Error::invalid(format!(
            "this table blob is version {}, and this build reads version {VERSION}",
            words[1]
        )));
    }
    let columns = usize::try_from(words[2])
        .map_err(|_| Error::invalid("a table blob declares more columns than fit a usize"))?;
    let rows = usize::try_from(words[3])
        .map_err(|_| Error::invalid("a table blob declares more rows than fit a usize"))?;
    // Checked before anything is sized from it. Every column costs at least two
    // words, so a count past the blob's own length is impossible — and reserving
    // for it first would turn a malformed blob into an allocation failure, which
    // is a crash where this module promises a refusal.
    if columns > words.len() {
        return Err(Error::invalid(format!(
            "a table blob of {} word(s) declares {columns} column(s), each of which is at least \
             two words",
            words.len()
        )));
    }

    let mut at = 4usize;
    let mut declared = Vec::with_capacity(columns);
    for index in 0..columns {
        if at + 2 > words.len() {
            return Err(Error::invalid(format!(
                "a table blob ends inside the header of column {index}"
            )));
        }
        let kind = ColumnType::from_code(words[at]).ok_or_else(|| {
            Error::invalid(format!(
                "column {index} of a table blob has type code {}, which this build does not know. \
                 Every column type is a compatibility surface, so an unknown one is refused \
                 rather than guessed at.",
                words[at]
            ))
        })?;
        let length = usize::try_from(words[at + 1])
            .map_err(|_| Error::invalid("a table blob declares an impossible column name"))?;
        at += 2;
        let name_words = length.div_ceil(8);
        if at + name_words > words.len() {
            return Err(Error::invalid(format!(
                "a table blob ends inside the name of column {index}"
            )));
        }
        let mut bytes = Vec::with_capacity(name_words * 8);
        for word in &words[at..at + name_words] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.truncate(length);
        let name = String::from_utf8(bytes).map_err(|_| {
            Error::invalid(format!(
                "the name of column {index} of a table blob is not UTF-8"
            ))
        })?;
        at += name_words;
        declared.push(Column::new(name, kind));
    }
    Ok((Schema::new(declared)?, rows, at))
}

// ------------------------------------------------------------------ table --

/// A row store, written per block and read by region.
///
/// See the module header for the two states, the canonical order, the schema and
/// why the index is not observable.
pub struct Table {
    volume: [usize; 3],
    schema: Arc<Schema>,
    /// What this table is called in its own error messages. `crate::points`
    /// builds one of these and needs its diagnostics to say "point store",
    /// because a caller who never asked for a table should not be told about
    /// one.
    noun: &'static str,
    /// Held while accumulating; taken by [`Self::seal`]. Decoded on the way in
    /// rather than at seal so that a malformed blob is refused while the block
    /// that wrote it is still known.
    pending: RowBuffer,
    /// One row of working space for [`Self::push`], reused; see there.
    scratch: Vec<u64>,
    index: Option<Box<dyn TableIndex>>,
    layout: Option<Layout>,
    /// Kept separately from `pending`, because `pending` is emptied by sealing
    /// and "how many rows does this table hold" must keep its answer.
    count: usize,
}

impl Table {
    /// An empty, accumulating table over `volume`.
    ///
    /// A volume with a zero-length axis holds no voxels, so no row could ever be
    /// inside it and no query could ever be answered; that is refused here
    /// rather than turned into a table that is silently always empty.
    pub fn new(volume: [usize; 3], schema: Schema) -> Result<Self> {
        Self::named(volume, schema, "table")
    }

    pub(crate) fn named(volume: [usize; 3], schema: Schema, noun: &'static str) -> Result<Self> {
        if volume.iter().any(|&length| length == 0) {
            return Err(Error::invalid(format!(
                "a {noun} over {volume:?} has no voxels for a row to be at, so every write would \
                 be refused and every query would answer nothing. A volume with a zero-length \
                 axis is a mistake upstream rather than an empty store."
            )));
        }
        let schema = Arc::new(schema);
        Ok(Self {
            volume,
            pending: RowBuffer::new(Arc::clone(&schema)),
            scratch: vec![0; schema.width()],
            schema,
            noun,
            index: None,
            layout: None,
            count: 0,
        })
    }

    pub fn volume(&self) -> [usize; 3] {
        self.volume
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// A builder for this table's schema.
    ///
    /// The producer-side path, and the reason it is here rather than only on
    /// [`RowBuilder`]: a builder made from the table it will be written to
    /// cannot have the wrong schema, which removes the commonest way to reach
    /// the mismatch error rather than merely reporting it well.
    pub fn builder(&self) -> RowBuilder {
        RowBuilder::new(Arc::clone(&self.schema))
    }

    pub fn state(&self) -> State {
        if self.index.is_some() {
            State::Sealed
        } else {
            State::Accumulating
        }
    }

    /// Which index was built, or `None` while accumulating.
    pub fn layout(&self) -> Option<Layout> {
        self.layout
    }

    /// How many rows have been written.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// One row, typed, straight into the table.
    ///
    /// The in-process path. Checks the schema exactly as [`RowBuilder::push`]
    /// does — it is the same function — and the volume as [`Self::write`] does.
    pub fn push(&mut self, at: [usize; 3], values: &[Value]) -> Result<()> {
        self.refuse_if_sealed(None, 0)?;
        if values.len() != self.schema.len() {
            return Err(Error::invalid(format!(
                "a row of this {} has {} value(s) — {} — and {} were given",
                self.noun,
                self.schema.len(),
                names(&self.schema),
                values.len()
            )));
        }
        for axis in 0..3 {
            if at[axis] >= self.volume[axis] {
                return Err(self.outside(None, self.count, at, axis));
            }
        }
        // Filled in scratch and copied in one go, so a value refused halfway
        // through leaves no half a row behind. Reused rather than allocated per
        // call because this is the path `crate::points` writes one point at a
        // time down.
        for axis in 0..3 {
            self.scratch[axis] = at[axis] as u64;
        }
        for (index, value) in values.iter().enumerate() {
            self.scratch[POSITION_WORDS + index] = checked_word(&self.schema, index, *value)?;
        }
        self.pending.push_words(&self.scratch);
        self.count += 1;
        Ok(())
    }

    /// Take one block's blob.
    ///
    /// **There is no ownership rule.** A block writes whatever rows it produced,
    /// wherever they landed, and nothing here asks whether they are near the
    /// block that wrote them — which is the only thing a block *can* do, since a
    /// `BlockOutput::fragment` is keyed by its own phase and its own block and
    /// there is no way to write into a neighbour's key. Which region eventually
    /// holds a row is decided by [`Self::seal`], from the coordinate.
    ///
    /// `block` is used for the error messages and for nothing else. The table
    /// keeps no record of where a row came from, which is what makes an order
    /// that tiebreaks on the source block unrepresentable rather than merely
    /// discouraged.
    ///
    /// Three things are refused rather than accepted quietly, and each is a blob
    /// that would otherwise read as something plausible: a **schema** that is not
    /// this table's, a **position** outside the volume — a coordinate no region
    /// of this table can name, so the row would be held and never returned — and
    /// a **non-finite float**, which the module header explains.
    pub fn write(&mut self, block: [usize; 3], bytes: &[u8]) -> Result<()> {
        self.refuse_if_sealed(Some(block), bytes.len())?;

        let words = unpack_u64(bytes).map_err(|err| self.from_block(block, err))?;
        let (schema, rows, start) =
            decode_header(&words).map_err(|err| self.from_block(block, err))?;
        self.schema
            .check_against(&schema, "the blob")
            .map_err(|err| self.from_block(block, err))?;

        let width = self.schema.width();
        let payload = &words[start..];
        // Divided rather than multiplied: a hostile header can declare a row
        // count whose product with the width does not fit a `usize`, and an
        // overflow here would be a wrapped comparison that a truncated blob
        // could pass.
        if payload.len() % width != 0 || payload.len() / width != rows {
            return Err(self.from_block(
                block,
                Error::invalid(format!(
                    "the blob declares {rows} row(s) of {width} word(s) each and carries {} word(s)",
                    payload.len()
                )),
            ));
        }

        // Every row is checked before any is kept: a refused write must not
        // leave half a blob behind, because "how many rows does this hold" would
        // then be a fact about where the failure was.
        for (index, row) in payload.chunks_exact(width).enumerate() {
            // Converted in full before any axis is judged, so that the message
            // names the position the row actually has rather than the part of
            // it that had been read when the check fired.
            let mut at = [0usize; 3];
            for axis in 0..3 {
                at[axis] = usize::try_from(row[axis]).map_err(|_| {
                    self.from_block(
                        block,
                        Error::invalid(format!(
                            "row {index} has coordinate {} on axis {axis}, which does not fit \
                             this platform's usize",
                            row[axis]
                        )),
                    )
                })?;
            }
            for axis in 0..3 {
                if at[axis] >= self.volume[axis] {
                    return Err(self.outside(Some(block), index, at, axis));
                }
            }
            for (column, definition) in self.schema.columns().iter().enumerate() {
                if definition.kind() == ColumnType::F64 {
                    let value = f64::from_bits(row[POSITION_WORDS + column]);
                    if !value.is_finite() {
                        return Err(self.from_block(
                            block,
                            Error::invalid(format!(
                                "row {index} has {value} in column {column}, named {:?}, which is \
                                 not finite. The canonical order tiebreaks on the column's bits \
                                 and any sum over the table would stop being a check on \
                                 anything, so it is refused here rather than carried.",
                                definition.name()
                            )),
                        ));
                    }
                }
            }
        }

        self.pending.extend_words(payload);
        self.count += rows;
        Ok(())
    }

    /// Sort what has been written and build an index for it, choosing from the
    /// row count.
    ///
    /// **This is a barrier.** Every writer must have finished; nothing here can
    /// check that, and the most it does is refuse the writes that arrive
    /// afterwards. The sort happens in one place, in this process — a
    /// distributed sort over a set too large for one node is the later problem
    /// and the module header says what it would take.
    pub fn seal(&mut self) -> Result<()> {
        let layout = if self.count <= FLAT_LIMIT {
            Layout::Flat
        } else {
            Layout::Gridded
        };
        self.seal_as(layout)
    }

    /// Seal with a named index rather than the derived one.
    ///
    /// Here so that the two can be built over the same input and compared, which
    /// is how "the choice is not observable" is asserted rather than asserted
    /// about. It is **not** a tuning knob: the two answer identically, so the
    /// only thing choosing differently from [`Self::seal`] can do is make a run
    /// slower.
    pub fn seal_as(&mut self, layout: Layout) -> Result<()> {
        if self.index.is_some() {
            return Err(Error::invalid(format!(
                "this {} is already {}. Sealing twice would either discard the index and rebuild \
                 it — work with no effect, since nothing can have been written since — or \
                 quietly mean something different from what it says.",
                self.noun,
                State::Sealed.as_str()
            )));
        }
        let rows = self.pending.take();
        let index: Box<dyn TableIndex> = match layout {
            Layout::Flat => Box::new(Flat::build(rows)),
            Layout::Gridded => Box::new(Gridded::build(self.volume, rows)),
        };
        self.index = Some(index);
        self.layout = Some(layout);
        Ok(())
    }

    /// A stream of every row in `region`, in the canonical order.
    ///
    /// **This is the read interface.** It borrows the table and the region and
    /// yields as it goes; at no point is the answer held, which is what makes a
    /// whole-volume read possible over a set that a second copy of would not
    /// fit. [`Self::query`] collects it, and is a convenience rather than the
    /// primitive.
    ///
    /// Half-open: a row at `region.start` is in the answer, one at
    /// `region.start + region.shape` is not.
    ///
    /// A region reaching outside the volume is refused rather than clipped. The
    /// table's domain is the volume; a question about somewhere outside it has
    /// no answer, and "nothing there" is the answer a genuinely empty region
    /// gets, so returning it would make the two indistinguishable.
    pub fn scan(&self, region: &Region) -> Result<Box<dyn Iterator<Item = Row<'_>> + '_>> {
        let Some(index) = self.index.as_ref() else {
            return Err(Error::invalid(format!(
                "this {} is {}, not {}, so it has no answer to give: {} row(s) have been written \
                 and no index has been built. An answer now would be a fact about which writers \
                 happened to have finished rather than about the row set, and an empty one would \
                 be indistinguishable from a region that really is empty. Call seal() once every \
                 writer has finished.",
                self.noun,
                State::Accumulating.as_str(),
                State::Sealed.as_str(),
                self.count
            )));
        };
        if region.ndim() != 3 {
            return Err(Error::invalid(format!(
                "a {} is queried with a 3-D region, got rank {}",
                self.noun,
                region.ndim()
            )));
        }
        region.check_within(&self.volume, &format!("{} query region", self.noun))?;
        Ok(index.scan(region))
    }

    /// [`Self::scan`], collected.
    ///
    /// Here because collecting is often what a caller wants and writing
    /// `.collect()` at every call site is noise. It is deliberately *not* the
    /// primitive: the allocation is now a decision visible where it is made,
    /// rather than something the interface imposes on a caller who only wanted
    /// to fold. What it allocates is one cursor per row and not one row per row
    /// — the values stay where they are — so even this is not a copy of the
    /// answer.
    pub fn query(&self, region: &Region) -> Result<Vec<Row<'_>>> {
        Ok(self.scan(region)?.collect())
    }

    fn refuse_if_sealed(&self, block: Option<[usize; 3]>, bytes: usize) -> Result<()> {
        if self.index.is_none() {
            return Ok(());
        }
        let who = match block {
            Some(block) => format!("block {block:?} tried to write {bytes} byte(s)"),
            None => "a row was pushed".to_string(),
        };
        Err(Error::invalid(format!(
            "this {} is {}, so it takes no more rows; {who}. Sealing is a barrier: the index was \
             built from everything that had been written, and a row arriving afterwards would be \
             in the table's data and not in its answers.",
            self.noun,
            State::Sealed.as_str()
        )))
    }

    fn outside(
        &self,
        block: Option<[usize; 3]>,
        index: usize,
        at: [usize; 3],
        axis: usize,
    ) -> Error {
        let who = match block {
            Some(block) => format!("row {index} of block {block:?}"),
            None => format!("row {index}"),
        };
        Error::invalid(format!(
            "{who} is at {at:?}, which is outside this {}'s volume {:?} on axis {axis}. A {} does \
             not care which block a row came from — that is the whole point of it — but a \
             coordinate outside the volume is one no query can ever name, so it would be kept and \
             never returned.",
            self.noun, self.volume, self.noun
        ))
    }

    fn from_block(&self, block: [usize; 3], err: Error) -> Error {
        Error::invalid(format!(
            "the blob written by block {block:?} is unusable for this {}: {err}",
            self.noun
        ))
    }
}

impl std::fmt::Debug for Table {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Table")
            .field("volume", &self.volume)
            .field("schema", &names(&self.schema))
            .field("state", &self.state().as_str())
            .field("rows", &self.count)
            .field("layout", &self.layout.map(Layout::as_str))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::detect::Moments;

    /// A SplitMix64 stream. The tests want *many* regions and *many* rows and
    /// want the same ones every run; a named constant seed is what makes a
    /// failure reproducible rather than a thing that happened once.
    ///
    /// Duplicated from `crate::points`' suite rather than shared, because that
    /// suite is the one this module must not disturb: a helper it imported from
    /// here would be a way for a change in this file to change what that file
    /// tests.
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, limit: usize) -> usize {
            (self.next() % limit as u64) as usize
        }
    }

    /// The schema the sweeps run over: an integer, a float, and a second integer
    /// after the float, so that a tiebreak that stopped at the first payload
    /// column would be caught.
    fn schema() -> Schema {
        Schema::new(vec![
            Column::u64("count"),
            Column::f64("total"),
            Column::u64("extent"),
        ])
        .unwrap()
    }

    /// A small pool of floats, so that rows tying on position with equal
    /// payloads and rows tying on position with different payloads both occur in
    /// a set this size.
    ///
    /// **Both zeros are in it.** `-0.0` and `0.0` are equal under `PartialEq`
    /// and differ in every bit that matters here: they are two distinct rows,
    /// they sort apart, and a comparison written on values rather than on bits
    /// would let two disagreeing answers pass as agreeing.
    const FLOATS: [f64; 5] = [-2.0, -1.0, -0.0, 0.0, 1.0];

    fn row_values(stream: &mut Stream) -> Vec<Value> {
        vec![
            Value::U64(stream.below(4) as u64),
            Value::F64(FLOATS[stream.below(FLOATS.len())]),
            Value::U64(stream.below(3) as u64),
        ]
    }

    fn scatter(volume: [usize; 3], count: usize, seed: u64) -> Vec<([usize; 3], Vec<Value>)> {
        let mut stream = Stream(seed);
        (0..count)
            .map(|_| {
                let at = [
                    stream.below(volume[0]),
                    stream.below(volume[1]),
                    stream.below(volume[2]),
                ];
                (at, row_values(&mut stream))
            })
            .collect()
    }

    /// The blobs a set of blocks would have written, with the rows scattered
    /// between them **without regard to where they are** — which is the write
    /// path this module exists to allow.
    fn blobs(
        schema: &Arc<Schema>,
        rows: &[([usize; 3], Vec<Value>)],
        blocks: usize,
        seed: u64,
    ) -> Vec<([usize; 3], Vec<u8>)> {
        let mut stream = Stream(seed);
        let mut builders: Vec<RowBuilder> = (0..blocks)
            .map(|_| RowBuilder::new(Arc::clone(schema)))
            .collect();
        for (at, values) in rows {
            let which = stream.below(blocks);
            builders[which].push(*at, values).unwrap();
        }
        builders
            .into_iter()
            .enumerate()
            .map(|(index, builder)| ([index, 0, 0], builder.encode()))
            .collect()
    }

    fn sealed(
        volume: [usize; 3],
        schema: Schema,
        written: &[([usize; 3], Vec<u8>)],
        layout: Layout,
    ) -> Table {
        let mut table = Table::new(volume, schema).unwrap();
        for (block, bytes) in written {
            table.write(*block, bytes).unwrap();
        }
        table.seal_as(layout).unwrap();
        table
    }

    /// Regions of every shape the interface has to survive: empty, one voxel,
    /// slabs, the whole volume, and boxes at the far corner.
    fn regions(volume: [usize; 3], seed: u64) -> Vec<Region> {
        let mut stream = Stream(seed);
        let half = [volume[0] / 2, volume[1] / 2, volume[2] / 2];
        let mut out = vec![
            Region::whole(&volume),
            // empty in three ways: no extent at all, and no extent on one axis
            Region::new(&[0, 0, 0], &[0, 0, 0]),
            Region::new(&[1, 1, 0], &[0, half[1], volume[2]]),
            Region::new(&[0, 0, 0], &[half[0], half[1], 0]),
            // one voxel, at the near corner and at the far one
            Region::new(&[0, 0, 0], &[1, 1, 1]),
            Region::new(&[volume[0] - 1, volume[1] - 1, volume[2] - 1], &[1, 1, 1]),
            // slabs along each axis
            Region::new(&[0, 0, 0], &[volume[0], 1, 1]),
            Region::new(&[0, 0, 0], &[1, volume[1], volume[2]]),
            // and the far octant, which is the one an off-by-one in the bucket
            // span reaches past
            Region::new(
                &half,
                &[
                    volume[0] - half[0],
                    volume[1] - half[1],
                    volume[2] - half[2],
                ],
            ),
        ];
        for _ in 0..200 {
            let mut start = [0usize; 3];
            let mut shape = [0usize; 3];
            for axis in 0..3 {
                start[axis] = stream.below(volume[axis]);
                shape[axis] = stream.below(volume[axis] - start[axis] + 1);
            }
            out.push(Region::new(&start, &shape));
        }
        out
    }

    // ------------------------------------------------------- the main one --

    /// **The most important test in the file.** If the two indexes ever answer
    /// differently, they are not two implementations of one thing and every
    /// claim about the choice being invisible is false.
    ///
    /// Compared **element by element off the two streams**, rather than by
    /// collecting them: a difference in position fails at that position, a
    /// difference in length fails as `Some` against `None`, and neither side is
    /// materialised to make the comparison.
    ///
    /// The comparison is on the row's **words**, which for a float column is its
    /// bits — so `-0.0` and `0.0` cannot pass as agreeing, and neither can two
    /// NaNs, which the write path refuses anyway.
    #[test]
    fn both_indexes_answer_every_query_identically() {
        for (volume, count, seed) in [
            ([40usize, 40, 40], 5_000usize, 11u64),
            // Also the degenerate end: few enough rows that the derivation
            // gives the gridded index a single bucket, which is the case where
            // an off-by-one in the span would otherwise hide.
            ([16, 12, 9], 12, 23),
            // And a flat volume, where an axis is clamped to its own length.
            ([64, 64, 1], 3_000, 37),
            // And a set with no rows at all.
            ([8, 8, 8], 0, 41),
        ] {
            let shared = Arc::new(schema());
            let rows = scatter(volume, count, seed);
            let written = blobs(&shared, &rows, 7, seed ^ 0xabc);
            let flat = sealed(volume, schema(), &written, Layout::Flat);
            let gridded = sealed(volume, schema(), &written, Layout::Gridded);
            assert_eq!(flat.len(), count);
            assert_eq!(gridded.len(), count);

            let asked = regions(volume, seed ^ 0xdef);
            let mut answered = 0usize;
            for region in &asked {
                let mut left = flat.scan(region).unwrap();
                let mut right = gridded.scan(region).unwrap();
                let mut at = 0usize;
                loop {
                    let from_flat = left.next().map(|row| row.words().to_vec());
                    let from_gridded = right.next().map(|row| row.words().to_vec());
                    assert_eq!(
                        from_flat, from_gridded,
                        "volume {volume:?}, {count} row(s), region {region:?}: the flat and \
                         gridded streams differ at position {at}"
                    );
                    if from_flat.is_none() {
                        break;
                    }
                    at += 1;
                    answered += 1;
                }
            }
            // Not vacuous: the sweep has to actually be finding rows, or two
            // empty answers would agree for the wrong reason.
            if count > 0 {
                assert!(
                    answered > count,
                    "the region sweep returned {answered} row(s) over {} region(s); it is not \
                     exercising the indexes",
                    asked.len()
                );
            }
        }
    }

    // ---------------------------------------------------------- the states --

    #[test]
    fn a_query_before_sealing_fails_and_names_the_state() {
        let mut table = Table::new([8, 8, 8], schema()).unwrap();
        assert_eq!(table.state(), State::Accumulating);
        assert_eq!(table.layout(), None);
        table
            .push([1, 1, 1], &[Value::U64(1), Value::F64(2.0), Value::U64(3)])
            .unwrap();

        let whole = Region::whole(&[8, 8, 8]);
        // Both the primitive and the convenience, because a stream that opened
        // and yielded nothing would be the quiet answer this refuses to give.
        for error in [
            table.scan(&whole).err().map(|err| err.to_string()).unwrap(),
            table.query(&whole).unwrap_err().to_string(),
        ] {
            assert!(error.contains("accumulating"), "{error}");
            assert!(error.contains("sealed"), "{error}");
            assert!(error.contains("seal()"), "{error}");
        }

        table.seal().unwrap();
        assert_eq!(table.state(), State::Sealed);
        assert_eq!(table.layout(), Some(Layout::Flat));
        assert_eq!(table.query(&whole).unwrap().len(), 1);
        assert_eq!(table.scan(&whole).unwrap().count(), 1);
    }

    #[test]
    fn a_sealed_table_takes_no_more_rows_and_cannot_be_sealed_twice() {
        let mut table = Table::new([8, 8, 8], schema()).unwrap();
        table.seal().unwrap();
        let builder_bytes = {
            let mut builder = table.builder();
            builder
                .push([0, 0, 0], &[Value::U64(0), Value::F64(0.0), Value::U64(0)])
                .unwrap();
            builder.encode()
        };
        let error = table
            .write([2, 0, 0], &builder_bytes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("sealed"), "{error}");
        assert!(error.contains("[2, 0, 0]"), "{error}");
        assert_eq!(table.len(), 0, "a refused write must not land");
        assert!(table
            .push([0, 0, 0], &[Value::U64(0), Value::F64(0.0), Value::U64(0)])
            .is_err());

        let error = table.seal().unwrap_err().to_string();
        assert!(error.contains("already"), "{error}");
        assert!(error.contains("sealed"), "{error}");
    }

    /// The sealer picks from the count, and both sides of the threshold are
    /// reachable. The *answers* are compared elsewhere.
    #[test]
    fn the_sealer_chooses_from_the_row_count() {
        let volume = [64usize, 64, 64];
        for (count, want) in [
            (FLAT_LIMIT, Layout::Flat),
            (FLAT_LIMIT + 1, Layout::Gridded),
        ] {
            let mut table = Table::new(volume, schema()).unwrap();
            for (at, values) in scatter(volume, count, 5) {
                table.push(at, &values).unwrap();
            }
            table.seal().unwrap();
            assert_eq!(table.layout(), Some(want));
        }
    }

    // ------------------------------------------------------- what is in it --

    /// Exactly the rows inside, and the half-open boundary stated in the header.
    /// Built by hand so the expected answer comes from the definition rather
    /// than from a second run of this code.
    #[test]
    fn a_query_returns_the_rows_inside_and_the_boundary_is_half_open() {
        let volume = [10usize, 10, 10];
        let placed = [
            [2usize, 2, 2], // the low corner of the region: in
            [5, 5, 5],      // the high corner of the region: out
            [4, 4, 4],      // strictly inside
            [1, 2, 2],      // one short on the first axis: out
            [2, 1, 2],      // out on the second
            [2, 2, 1],      // out on the third
            [5, 4, 4],      // on the far face of the first axis: out
            [4, 5, 4],      // the second: out
            [4, 4, 5],      // the third: out
        ];
        let rows: Vec<([usize; 3], Vec<Value>)> = placed
            .iter()
            .map(|&at| (at, vec![Value::U64(1), Value::F64(1.0), Value::U64(1)]))
            .collect();
        let shared = Arc::new(schema());
        let written = blobs(&shared, &rows, 3, 99);
        for layout in [Layout::Flat, Layout::Gridded] {
            let table = sealed(volume, schema(), &written, layout);
            let found: Vec<[usize; 3]> = table
                .query(&Region::new(&[2, 2, 2], &[3, 3, 3]))
                .unwrap()
                .iter()
                .map(Row::at)
                .collect();
            assert_eq!(found, vec![[2, 2, 2], [4, 4, 4]], "{}", layout.as_str());
            // A zero-shape region holds nothing, including the row at its own
            // start.
            assert!(table
                .query(&Region::new(&[2, 2, 2], &[0, 3, 3]))
                .unwrap()
                .is_empty());
            // And the whole volume holds everything.
            assert_eq!(table.query(&Region::whole(&volume)).unwrap().len(), 9);
        }
    }

    #[test]
    fn a_region_outside_the_volume_or_of_the_wrong_rank_is_refused() {
        let table = sealed([8, 8, 8], schema(), &[], Layout::Flat);
        let error = table
            .query(&Region::new(&[6, 0, 0], &[4, 1, 1]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("table query region"), "{error}");
        let error = table
            .query(&Region::new(&[0, 0], &[1, 1]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("rank 2"), "{error}");
    }

    #[test]
    fn a_row_outside_the_volume_is_refused_and_says_it_is_not_about_ownership() {
        let mut table = Table::new([8, 4, 4], schema()).unwrap();
        let mut builder = table.builder();
        builder
            .push([8, 0, 0], &[Value::U64(1), Value::F64(1.0), Value::U64(1)])
            .unwrap();
        // Deliberately keyed to a block nowhere near the row: that part is fine
        // here, and the message has to say so.
        let error = table
            .write([0, 0, 0], &builder.encode())
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside this table's volume"), "{error}");
        assert!(error.contains("does not care which block"), "{error}");
        assert_eq!(table.len(), 0);
        assert!(Table::new([8, 0, 4], schema()).is_err());
    }

    // -------------------------------------------------------- the ordering --

    /// Shuffle the blobs, seal again, and the words must not move.
    ///
    /// The set deliberately contains rows that **tie on the position columns and
    /// differ in a later one**, so the tiebreak is exercised rather than vacuous
    /// — and the assertion at the end pins that, by checking the order the three
    /// coincident rows come back in against the order they were written in.
    /// Two of them agree on the first payload column and differ only on the
    /// third, so a tiebreak that stopped after one column would fail here.
    #[test]
    fn the_canonical_order_survives_a_shuffle_of_the_blobs() {
        let volume = [24usize, 24, 24];
        let mut rows = scatter(volume, 400, 101);
        // Three rows at one coordinate, written in an order that is not their
        // canonical one.
        rows.push((
            [7, 7, 7],
            vec![Value::U64(2), Value::F64(1.0), Value::U64(9)],
        ));
        rows.push((
            [7, 7, 7],
            vec![Value::U64(2), Value::F64(1.0), Value::U64(4)],
        ));
        rows.push((
            [7, 7, 7],
            vec![Value::U64(1), Value::F64(8.0), Value::U64(7)],
        ));

        let shared = Arc::new(schema());
        let mut written = blobs(&shared, &rows, 9, 202);
        let reference = sealed(volume, schema(), &written, Layout::Flat);
        let expected: Vec<Vec<u64>> = reference
            .query(&Region::whole(&volume))
            .unwrap()
            .iter()
            .map(|row| row.words().to_vec())
            .collect();

        let mut stream = Stream(303);
        for round in 0..8 {
            // Fisher-Yates over the blobs, which is the only freedom a caller
            // has: which block wrote a row, and in which order the blobs arrive.
            for index in (1..written.len()).rev() {
                written.swap(index, stream.below(index + 1));
            }
            for layout in [Layout::Flat, Layout::Gridded] {
                let table = sealed(volume, schema(), &written, layout);
                let got: Vec<Vec<u64>> = table
                    .query(&Region::whole(&volume))
                    .unwrap()
                    .iter()
                    .map(|row| row.words().to_vec())
                    .collect();
                assert_eq!(
                    got,
                    expected,
                    "round {round}, {}: the order moved when the blobs did",
                    layout.as_str()
                );
            }
        }

        // The tiebreak, pinned. Canonically: (1, 8.0, 7) first because its
        // first payload column is smaller, then (2, 1.0, 4) before (2, 1.0, 9)
        // — which is decided by the **third** column, so a tiebreak that
        // stopped at the first would order these two arbitrarily.
        let coincident: Vec<(u64, f64, u64)> = reference
            .query(&Region::new(&[7, 7, 7], &[1, 1, 1]))
            .unwrap()
            .iter()
            .filter(|row| row.at() == [7, 7, 7])
            .map(|row| {
                (
                    row.u64(0).unwrap(),
                    row.f64(1).unwrap(),
                    row.u64(2).unwrap(),
                )
            })
            .collect();
        assert_eq!(
            coincident[coincident.len() - 3..],
            [(1, 8.0, 7), (2, 1.0, 4), (2, 1.0, 9)],
            "the tiebreak does not walk every payload column"
        );
    }

    /// The table's own promise: what comes back is in the canonical order, not
    /// merely the same order twice.
    ///
    /// The oracle is written from the definition in the header — position
    /// triple, then the payload columns' bits — rather than by calling
    /// [`canonical`], so a change to that function does not silently change what
    /// this asserts.
    #[test]
    fn a_query_answers_in_the_canonical_order() {
        let volume = [30usize, 30, 30];
        let shared = Arc::new(schema());
        let rows = scatter(volume, 6_000, 13);
        let written = blobs(&shared, &rows, 11, 14);
        for layout in [Layout::Flat, Layout::Gridded] {
            let table = sealed(volume, schema(), &written, layout);
            for region in regions(volume, 15).iter().take(30) {
                let found = table.query(region).unwrap();
                for pair in found.windows(2) {
                    let (left, right) = (pair[0], pair[1]);
                    let key = |row: Row<'_>| {
                        let mut out: Vec<u64> = row.at().iter().map(|&v| v as u64).collect();
                        for column in 0..row.schema().len() {
                            out.push(match row.value(column).unwrap() {
                                Value::U64(value) => value,
                                Value::F64(value) => value.to_bits(),
                            });
                        }
                        out
                    };
                    assert!(
                        key(left) <= key(right),
                        "{}: {left:?} came back before {right:?}",
                        layout.as_str()
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------- the schema --

    /// A wrong schema is refused where the bytes arrive, and the message names
    /// the column and both types.
    #[test]
    fn a_mismatched_schema_is_refused_at_write_time_naming_the_column() {
        let volume = [8usize, 8, 8];
        let mut table = Table::new(volume, schema()).unwrap();

        // The right names in the right order, with one type changed.
        let wrong_type = Arc::new(
            Schema::new(vec![
                Column::u64("count"),
                Column::u64("total"),
                Column::u64("extent"),
            ])
            .unwrap(),
        );
        let mut builder = RowBuilder::new(Arc::clone(&wrong_type));
        builder
            .push([0, 0, 0], &[Value::U64(1), Value::U64(2), Value::U64(3)])
            .unwrap();
        let error = table
            .write([0, 0, 0], &builder.encode())
            .unwrap_err()
            .to_string();
        assert!(error.contains("column 1"), "{error}");
        assert!(error.contains("total"), "{error}");
        assert!(error.contains("u64"), "{error}");
        assert!(error.contains("f64"), "{error}");
        assert_eq!(table.len(), 0);

        // The right types under a different name.
        let renamed = Arc::new(
            Schema::new(vec![
                Column::u64("count"),
                Column::f64("sum"),
                Column::u64("extent"),
            ])
            .unwrap(),
        );
        let mut builder = RowBuilder::new(renamed);
        builder
            .push([0, 0, 0], &[Value::U64(1), Value::F64(2.0), Value::U64(3)])
            .unwrap();
        let error = table
            .write([0, 0, 0], &builder.encode())
            .unwrap_err()
            .to_string();
        assert!(error.contains("sum"), "{error}");
        assert!(error.contains("total"), "{error}");

        // Too few columns.
        let short = Arc::new(Schema::new(vec![Column::u64("count")]).unwrap());
        let mut builder = RowBuilder::new(short);
        builder.push([0, 0, 0], &[Value::U64(1)]).unwrap();
        let error = table
            .write([0, 0, 0], &builder.encode())
            .unwrap_err()
            .to_string();
        assert!(error.contains("1 column(s)"), "{error}");
        assert!(error.contains("3"), "{error}");

        // And a value of the wrong type, refused where it is supplied.
        let error = table
            .push([0, 0, 0], &[Value::U64(1), Value::U64(2), Value::U64(3)])
            .unwrap_err()
            .to_string();
        assert!(error.contains("column 1"), "{error}");
        assert!(error.contains("total"), "{error}");
        assert!(error.contains("is f64"), "{error}");
        assert!(error.contains("a u64 was given"), "{error}");
        // The wrong number of them, too.
        assert!(table.push([0, 0, 0], &[Value::U64(1)]).is_err());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn a_schema_refuses_an_unnamed_or_repeated_column() {
        assert!(Schema::new(vec![Column::u64("")]).is_err());
        let error = Schema::new(vec![Column::u64("count"), Column::f64("count")])
            .unwrap_err()
            .to_string();
        assert!(error.contains("count"), "{error}");
        assert!(error.contains("ambiguous"), "{error}");
        // A table with no payload columns at all is a pure position set, and is
        // allowed: the canonical order is then the position alone and rows that
        // tie on it are genuinely indistinguishable.
        let positions = Schema::new(Vec::new()).unwrap();
        assert_eq!(positions.width(), POSITION_WORDS);
        let mut table = Table::new([4, 4, 4], positions).unwrap();
        table.push([1, 2, 3], &[]).unwrap();
        table.push([1, 2, 3], &[]).unwrap();
        table.seal().unwrap();
        assert_eq!(table.query(&Region::whole(&[4, 4, 4])).unwrap().len(), 2);
    }

    #[test]
    fn a_non_finite_float_is_refused_naming_the_column() {
        let mut table = Table::new([8, 8, 8], schema()).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = table
                .push([0, 0, 0], &[Value::U64(1), Value::F64(bad), Value::U64(1)])
                .unwrap_err()
                .to_string();
            assert!(error.contains("total"), "{error}");
            assert!(error.contains("not finite"), "{error}");
        }
        // And through the blob, where the builder is the one that refused it,
        // so the check has to be repeated on the way in — a blob can arrive
        // from anywhere.
        let mut words = vec![MAGIC, VERSION, 3, 1];
        for (code, name) in [(1u64, "count"), (2, "total"), (1, "extent")] {
            words.push(code);
            words.push(name.len() as u64);
            for chunk in name.as_bytes().chunks(8) {
                let mut padded = [0u8; 8];
                padded[..chunk.len()].copy_from_slice(chunk);
                words.push(u64::from_le_bytes(padded));
            }
        }
        words.extend_from_slice(&[0, 0, 0, 1, f64::NAN.to_bits(), 1]);
        let error = table
            .write([4, 5, 6], &pack_u64(&words))
            .unwrap_err()
            .to_string();
        assert!(error.contains("not finite"), "{error}");
        assert!(error.contains("total"), "{error}");
        assert!(error.contains("[4, 5, 6]"), "{error}");
    }

    // -------------------------------------------------------- the encoding --

    #[test]
    fn encoding_round_trips_and_a_broken_blob_is_refused() {
        let shared = Arc::new(schema());
        let mut builder = RowBuilder::new(Arc::clone(&shared));
        builder
            .push([0, 0, 0], &[Value::U64(0), Value::F64(-2.5), Value::U64(7)])
            .unwrap();
        builder
            .push(
                [3, 1, 4],
                &[
                    Value::U64(u64::MAX),
                    Value::F64(f64::MIN_POSITIVE),
                    Value::U64(0),
                ],
            )
            .unwrap();
        let bytes = builder.encode();

        // The blob says what it is, without a schema in hand.
        assert_eq!(encoded_schema(&bytes).unwrap(), *shared);

        let mut table = Table::new([8, 8, 8], schema()).unwrap();
        table.write([0, 0, 0], &bytes).unwrap();
        table.seal().unwrap();
        let found = table.query(&Region::whole(&[8, 8, 8])).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].at(), [0, 0, 0]);
        assert_eq!(found[0].u64(0).unwrap(), 0);
        assert_eq!(found[0].f64(1).unwrap(), -2.5);
        assert_eq!(
            found[1].u64(0).unwrap(),
            u64::MAX,
            "a wide integer survives"
        );
        assert_eq!(found[1].f64(1).unwrap(), f64::MIN_POSITIVE);
        assert_eq!(found[1].by_name("extent").unwrap(), Value::U64(0));
        assert!(found[0].u64(1).is_err(), "a float read as an integer");
        assert!(found[0].f64(0).is_err(), "an integer read as a float");
        assert!(found[0].value(3).is_err(), "a column that is not there");
        assert!(found[0].by_name("nothing").is_err());

        let mut fresh = Table::new([8, 8, 8], schema()).unwrap();
        // Not words at all.
        assert!(fresh.write([1, 2, 3], &bytes[..9]).is_err());
        // Words, but not a table.
        let error = fresh
            .write([1, 2, 3], &pack_u64(&[1, 2, 3, 4]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("magic"), "{error}");
        assert!(error.contains("[1, 2, 3]"), "{error}");
        // A table, truncated inside its rows.
        let error = fresh
            .write([0, 0, 0], &bytes[..bytes.len() - 8])
            .unwrap_err()
            .to_string();
        assert!(error.contains("row(s)"), "{error}");
        // A table with something appended.
        let mut extra = bytes.clone();
        extra.extend_from_slice(&[0; 8]);
        assert!(fresh.write([0, 0, 0], &extra).is_err());
        // A version this build does not read.
        let mut wrong_version = unpack_u64(&bytes).unwrap();
        wrong_version[1] = VERSION + 1;
        assert!(fresh.write([0, 0, 0], &pack_u64(&wrong_version)).is_err());
        // A column type code this build does not know.
        let mut wrong_code = unpack_u64(&bytes).unwrap();
        wrong_code[4] = 99;
        let error = fresh
            .write([0, 0, 0], &pack_u64(&wrong_code))
            .unwrap_err()
            .to_string();
        assert!(error.contains("type code 99"), "{error}");
        assert_eq!(fresh.len(), 0, "no refused blob landed");

        // An empty blob is a block with nothing to say, which is a fact and not
        // a failure.
        let empty = RowBuilder::new(shared).encode();
        fresh.write([9, 9, 9], &empty).unwrap();
        assert_eq!(fresh.len(), 0);
    }

    // ------------------------------------------------------- the exactness --

    /// **What the `U64` column is for.** `ops::detect` accumulates a component's
    /// voxel count and position sums as integers, and its header says why: an
    /// `f64` first moment stops being exact at a volume edge of about 11500
    /// voxels. A table that could only carry the answer through a float column
    /// would have given that back in transport.
    ///
    /// So: split an accumulation four ways, merge it in three different orders
    /// with `ops::detect`'s own merge, put the total through a table, and read
    /// it back — the same words as the same accumulation seen whole, exactly.
    /// The magnitudes are chosen above `2^53`, and the last assertion shows the
    /// `f64` path demonstrably losing them, so this is not vacuous.
    #[test]
    fn a_u64_column_carries_a_split_accumulation_exactly() {
        // A component whose first moment is past the exact range of an f64.
        let base = (1u64 << 53) + 1;
        let voxels: Vec<[usize; 3]> = (0..8)
            .map(|step| [base as usize + step, 2 + step, 5])
            .collect();

        let mut whole = Moments::EMPTY;
        for &at in &voxels {
            whole.add(at).unwrap();
        }

        // Four pieces, as four blocks would have produced them, merged in three
        // orders. `Moments::merge` is integer addition, so all three agree.
        let mut pieces = Vec::new();
        for slice in voxels.chunks(2) {
            let mut part = Moments::EMPTY;
            for &at in slice {
                part.add(at).unwrap();
            }
            pieces.push(part);
        }
        let table_schema = Schema::new(vec![
            Column::u64("count"),
            Column::u64("sum_0"),
            Column::u64("sum_1"),
            Column::u64("sum_2"),
        ])
        .unwrap();
        let volume = [base as usize + 16, 16, 16];

        let row_of = |moments: &Moments| {
            let at = moments.centroid().unwrap();
            (
                at,
                vec![
                    Value::U64(moments.count),
                    Value::U64(moments.sums[0]),
                    Value::U64(moments.sums[1]),
                    Value::U64(moments.sums[2]),
                ],
            )
        };

        let mut reference = Table::new(volume, table_schema.clone()).unwrap();
        let (at, values) = row_of(&whole);
        reference.push(at, &values).unwrap();
        reference.seal().unwrap();
        let want: Vec<u64> = reference
            .query(&Region::whole(&volume))
            .unwrap()
            .first()
            .unwrap()
            .words()
            .to_vec();

        for order in [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]] {
            let mut merged = Moments::EMPTY;
            for slot in order {
                merged.merge(&pieces[slot]).unwrap();
            }
            assert_eq!(merged, whole, "order {order:?}");

            // Through a blob and back, which is the path a real run takes.
            let mut split = Table::new(volume, table_schema.clone()).unwrap();
            let mut builder = split.builder();
            let (at, values) = row_of(&merged);
            builder.push(at, &values).unwrap();
            split.write([1, 1, 1], &builder.encode()).unwrap();
            split.seal_as(Layout::Gridded).unwrap();
            let got: Vec<u64> = split
                .query(&Region::whole(&volume))
                .unwrap()
                .first()
                .unwrap()
                .words()
                .to_vec();
            assert_eq!(got, want, "order {order:?}: the table moved the answer");
        }

        // Not vacuous: the same numbers through an f64 column would not come
        // back. This is the assertion that says the type set had to have an
        // exact integer in it.
        assert_ne!(
            whole.sums[0] as f64 as u64, whole.sums[0],
            "the magnitudes in this test are inside f64's exact range, so it proves nothing"
        );
    }

    /// The other half of the exactness statement, and it is an honest one: an
    /// `F64` column round trips bit for bit, and folding one is still
    /// order-dependent. The store fixes the *order*; it cannot make `+`
    /// associate.
    #[test]
    fn a_float_column_round_trips_exactly_and_still_does_not_associate() {
        let volume = [8usize, 8, 8];
        let one_column = Schema::new(vec![Column::f64("total")]).unwrap();
        let values = [1.0, 1e-16, 1e-16, -0.0];
        let mut table = Table::new(volume, one_column).unwrap();
        for (index, &value) in values.iter().enumerate() {
            table.push([index, 0, 0], &[Value::F64(value)]).unwrap();
        }
        table.seal().unwrap();
        let found = table.query(&Region::whole(&volume)).unwrap();
        for (row, &value) in found.iter().zip(values.iter()) {
            // On bits, so that -0.0 does not pass for 0.0.
            assert_eq!(row.f64(0).unwrap().to_bits(), value.to_bits());
        }

        // And the reason `ops::detect` does not accumulate in one of these: the
        // same three numbers added in two orders are two different numbers.
        let forwards = (1.0f64 + 1e-16) + 1e-16;
        let backwards = 1.0f64 + (1e-16 + 1e-16);
        assert_ne!(forwards.to_bits(), backwards.to_bits());
    }

    // ------------------------------------------------- the point special case --

    /// The unification, asserted rather than described: a point blob is a row of
    /// the point schema, word for word.
    ///
    /// If this ever fails, `PointStore` is no longer a `Table` over
    /// [`Schema::points`] and the module header's claim that no existing answer
    /// moved is false.
    #[test]
    fn a_point_blob_is_a_row_of_the_point_schema() {
        use crate::points::{encode_points, Point};

        let points = [
            Point::unit([0, 0, 0]),
            Point::weighted([3, 1, 4], -2.5),
            Point::weighted([7, 8, 9], 1e300),
        ];
        let shared = Arc::new(Schema::points());
        let mut builder = RowBuilder::new(shared);
        for point in &points {
            builder.push(point.at, &[Value::F64(point.weight)]).unwrap();
        }
        // The row words and the point blob's words are the same sequence. They
        // are not the same *bytes*, because a table blob carries its schema in
        // front — which is the whole difference between the two channels.
        assert_eq!(
            builder.rows.words,
            unpack_u64(&encode_points(&points)).unwrap()
        );
    }

    // --------------------------------------------------------- the buckets --
    //
    // The index's own structural claims — that a small region reads a small
    // part of the index, that the bucket derivation lands near its target
    // occupancy, and that the merge's heap never exceeds one grid cross-section
    // — are asserted in `crate::points`' suite, over this `Gridded` through the
    // point payload. They are not repeated here. Duplicating them would assert
    // the same code twice and hide, rather than demonstrate, that the point
    // store and the table share one index; the sweep above already drives the
    // gridded path over a six-word row, which is what a wider schema could
    // break.

    /// The permutation the sort applies, on its own: a cycle-following in-place
    /// rearrangement is easy to get subtly wrong, and a wrong one would show up
    /// as a scrambled table rather than as a crash.
    #[test]
    fn the_in_place_permutation_is_a_permutation() {
        let shared = Arc::new(Schema::new(vec![Column::u64("k")]).unwrap());
        let mut stream = Stream(7);
        for rows in [0usize, 1, 2, 3, 5, 17, 64] {
            let mut buffer = RowBuffer::new(Arc::clone(&shared));
            let mut expected: Vec<Vec<u64>> = Vec::new();
            for _ in 0..rows {
                let words = vec![
                    stream.below(8) as u64,
                    stream.below(8) as u64,
                    stream.below(8) as u64,
                    stream.next(),
                ];
                expected.push(words.clone());
                buffer.push_words(&words);
            }
            buffer.sort_rows(canonical);
            expected.sort();
            let got: Vec<Vec<u64>> = (0..buffer.len())
                .map(|row| buffer.words_of(row).to_vec())
                .collect();
            assert_eq!(got, expected, "{rows} row(s)");
        }
    }
}
