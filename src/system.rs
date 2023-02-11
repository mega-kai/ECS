use crate::{command_buffer::*, component::*, query::*};

#[derive(Debug, Clone, Copy)]
pub enum ExecutionFrequency {
    Always,
    Once,
    //make sure it's not smaller than the tick duration
    Timed(f64), //in sec
}

/// a system containing metadata
#[derive(Clone, Copy, Debug)]
pub struct System {
    //add ordering information
    execution_frequency: ExecutionFrequency,
    //no dynamic querying
    pub(crate) fn_ptr: fn(Command, ()) -> Command,
}
impl System {
    /// generate a stub system that runs every game tick
    pub fn default(fn_ptr: fn(Command, ()) -> Command) -> Self {
        Self {
            fn_ptr,
            execution_frequency: ExecutionFrequency::Always,
        }
    }
}

pub trait SysParam {}
impl SysParam for Command {}
impl<C: Component> SysParam for Query<C> {}
impl<S1> SysParam for (S1,) where S1: SysParam {}
impl<S1, S2> SysParam for (S1, S2)
where
    S1: SysParam,
    S2: SysParam,
{
}

pub trait SysFn<SP: SysParam> {}
impl<Func, P0> SysFn<(P0,)> for Func
where
    P0: SysParam,
    Func: FnMut(P0) -> Command,
{
}

impl<Func, P0, P1> SysFn<(P0, P1)> for Func
where
    P0: SysParam,
    P1: SysParam,
    Func: FnMut(P0, P1) -> Command,
{
}
