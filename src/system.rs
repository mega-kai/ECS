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
pub struct SystemMetadata {
    execution_frequency: ExecutionFrequency,
}
impl SystemMetadata {
    pub fn default() -> Self {
        Self {
            execution_frequency: ExecutionFrequency::Always,
        }
    }
}

pub trait SysParam {}
impl SysParam for Command {}
//impl SysParam for Query {}
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
