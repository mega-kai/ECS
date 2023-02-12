use std::{default, marker::PhantomData};

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
    pub fn new(frequency: ExecutionFrequency) -> Self {
        Self {
            execution_frequency: frequency,
        }
    }
}

pub trait SysParam {}
impl SysParam for () {}
impl<S1> SysParam for (S1,) where S1: SysParam {}
impl<S1, S2> SysParam for (S1, S2)
where
    S1: SysParam,
    S2: SysParam,
{
}

pub trait SysFn<SP: SysParam> {
    fn run(&mut self);
}
impl<Func> SysFn<()> for Func
where
    Func: FnMut(),
{
    fn run(&mut self) {
        (self)()
    }
}
impl<Func, P0> SysFn<(P0,)> for Func
where
    P0: SysParam,
    Func: FnMut(P0) -> Command,
{
    fn run(&mut self) {
        //(self)()
    }
}
impl<Func, P0, P1> SysFn<(P0, P1)> for Func
where
    P0: SysParam,
    P1: SysParam,
    Func: FnMut(P0, P1) -> Command,
{
    fn run(&mut self) {
        //(self)()
    }
}

pub struct SystemWithMetadata<Param: SysParam, Sys: SysFn<Param>> {
    meta: SystemMetadata,
    func: Sys,
    /// just to bound the damn thing
    _marker: PhantomData<Param>,
}
impl<Param: SysParam, Sys: SysFn<Param>> SystemWithMetadata<Param, Sys> {
    pub fn default(func: Sys) -> Self {
        Self {
            meta: SystemMetadata::default(),
            func,
            _marker: PhantomData,
        }
    }
    pub fn once(func: Sys) -> Self {
        Self {
            meta: SystemMetadata::new(ExecutionFrequency::Once),
            func,
            _marker: PhantomData,
        }
    }
    pub fn new(func: Sys, meta: SystemMetadata) -> Self {
        Self {
            meta,
            func,
            _marker: PhantomData,
        }
    }
}

/// full system func along with its metadata
pub trait System {
    fn run(&mut self, input: ());
    fn yield_metadata(&self) -> SystemMetadata;
}
impl<Param: SysParam, Sys: SysFn<Param>> System for SystemWithMetadata<Param, Sys> {
    fn run(&mut self, input: ()) {
        self.func.run();
    }

    fn yield_metadata(&self) -> SystemMetadata {
        self.meta
    }
}
