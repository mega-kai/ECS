// the query api

// queying a single component by an id or index is not allowed;

// filtering, unique filtering creates archetype

// a distinction between (&mut A, &B) and (&mut A, With<B>) is that
// the latter does not issue a shared reference iterator of B, thus in
// the same cycle an iterator of mutable refs can still be issued, as
// With only guarantees that such component exists along with the
// ones you are querying, while &B can give you read access to its data

// only thing about query result is that it needs to be pregenerated before
// execution phase, and only changed when the storage/system pool changed

use std::marker::PhantomData;

use crate::component::*;

/// component marker
pub trait QueryMarker {}
impl<T> QueryMarker for &T where T: Component {}
impl<T> QueryMarker for &mut T where T: Component {}

/// query request, which also can be turned
/// into an iterator to get the access
#[derive(Debug)]
pub struct Query {
    /// consisted of multiple types
    access: (),
    filter: (),
}
impl Query {
    pub fn new() {}
}

pub struct TestQuery<T, U> {
    _1: PhantomData<T>,
    _2: PhantomData<U>,
}
impl<T, U> TestQuery<T, U> {
    fn new<A, B>() -> Self {
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
struct Health(i32);
impl Component for Health {}

#[derive(Debug, Clone, Copy)]
struct Mana(i32);
impl Component for Mana {}

#[derive(Debug, Clone, Copy)]
struct Player(&'static str);
impl Component for Player {}

fn test() {}
