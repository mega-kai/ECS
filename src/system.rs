use std::{default, marker::PhantomData};

#[derive(Debug, Clone, Copy)]
pub enum ExecutionFrequency {
    Always,
    Once,
    //make sure it's not smaller than the tick duration
    Timed(f64, f64), //in sec
}
