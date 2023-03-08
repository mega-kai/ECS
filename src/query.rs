use std::marker::PhantomData;

use crate::{component::*, storage::*, system::*};

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

#[derive(Debug)]
pub struct Query<Access = (), Filter = ()>
where
    Access: QueryMarker,
    Filter: QueryFilter,
{
    access: Access,
    filter: Filter,
}

impl<Access: QueryMarker, Filter: QueryFilter> SysParam for Query<Access, Filter> {}

pub struct QueryResult<Q: QueryMarker> {
    _marker: PhantomData<Q>,
}
