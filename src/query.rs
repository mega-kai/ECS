//the query api

// queying a single component by an id or index is not allowed;

// filtering, unique filtering creates archetype

// a distinction between (&mut A, &B) and (&mut A, With<B>) is that
// the latter does not issue a shared reference iterator of B, thus in
// the same cycle an iterator of mutable refs can still be issued, as
// With only guarantees that such component exists along with the
// ones you are querying, while &B can give you read access to its data

// only thing about query result is that it needs to be pregenerated before
// execution phase, and only changed when the storage/system pool changed

use crate::component::*;

#[derive(Debug)]
pub struct Query<C: Component> {
    //should also be an intoiterator, sorting the data
    access: Vec<C>,
    filter: (),
}
impl<C: Component> Query<C> {
    fn new() -> Self {
        todo!()
    }
}
