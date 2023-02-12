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

use crate::{component::*, system::*};

/// access
pub trait QueryMarker {}
impl QueryMarker for () {}
impl<T> QueryMarker for &T where T: Component {}
impl<T> QueryMarker for &mut T where T: Component {}

/// filters
pub struct With<Q: QueryMarker> {
    _data: PhantomData<Q>,
}
pub struct Without<Q: QueryMarker> {
    _data: PhantomData<Q>,
}

pub trait QueryFilter {}
impl QueryFilter for () {}
impl<Q: QueryMarker> QueryFilter for With<Q> {}
impl<Q: QueryMarker> QueryFilter for Without<Q> {}

/// query request, which also can be turned
/// into an iterator to get the access
#[derive(Debug)]
pub struct Query<Access = (), Filter = ()>
where
    Access: QueryMarker,
    Filter: QueryFilter,
{
    /// consisted of multiple types
    access: Access,
    filter: Filter,
}
/// make this a valid system parameter
impl<Access: QueryMarker, Filter: QueryFilter> SysParam for Query<Access, Filter> {}
