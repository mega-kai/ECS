//the query api

// FOR QUERYING
//  return an iterator of &C/&mut C, querying a
// single component by an id or index is not allowed;
//
// should also be able to query for (&mut A, ..., With/Or/Without<&B>, ...)
// which are basically archetypes;
//
// a distinction between (&mut A, &B) and (&mut A, With<B>) is that
// the latter does not issue a shared reference iterator of B, thus in
// the same cycle an iterator of mutable refs can still be issued, as
// With only guarantees that such component exists along with the
// ones you are querying, while &B can give you read access to its data

/// a list of desired references with filter functionalities;
/// part of the metadata, as with all other metadatas, it will be
/// processed by the scheduler
#[derive(Debug, Clone, Copy)]
pub struct QueryList {}
impl QueryList {
    pub fn empty() -> Self {
        Self {}
    }
}

/// a single query result that can be iterated upon to get all the references
/// to the components/entities
pub struct QueryResult(/* a bunch of references to components */);
impl QueryResult {
    pub fn new() -> Self {
        todo!()
    }

    pub fn does_things_with_the_result(&mut self) {}
}
