#![allow(
    dead_code,
    unused_variables,
    unreachable_code,
    unused_mut,
    unused_assignments,
    unused_imports
)]
#![feature(alloc_layout_extra, core_intrinsics, portable_simd)]

// todo, thread safety, meaning all that access to anything within table shouldn't cause data race
// preferably using atomics???

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::alloc::{alloc, dealloc, realloc};
use std::collections::HashMap;
use std::intrinsics::type_id;
use std::marker::PhantomData;
use std::mem::{forget, MaybeUninit};
use std::num::Wrapping;
use std::ops::{BitAnd, BitOr, BitXor, Deref, DerefMut, Not, Range};
use std::ptr::copy;
use std::simd::Simd;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::{alloc::Layout, fmt::Debug};

type SparseIndex = usize;
type DenseIndex = usize;
type BufferIndex = usize;
type TypeId = u128;

const MASK_HEAD: usize = 1 << (usize::BITS - 1);
const MASK_TAIL: usize = !MASK_HEAD;

const SIMD_START: Simd<usize, 64> = Simd::from_array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]);

const SHIFT: Simd<usize, 64> = Simd::from_array([63; 64]);
const ONE: Simd<usize, 64> = Simd::from_array([1; 64]);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BufferIndx {
    ONES,
    ZEROS,
    Index(BufferIndex),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operation {
    And,
    Or,
    Xor,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Node {
    // index in ids
    Single(bool, u8),
    Dual(bool, u8, Operation, u8),
}

impl Debug for Filter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Filter")
            .field("num_of_nodes", &self.num_of_nodes)
            .field("num_of_ids", &self.num_of_ids)
            .field("nodes", &self.nodes)
            // .field("ids", &self.ids)
            .finish()
    }
}

const NUM_OF_NODES: usize = 16;
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Filter {
    num_of_nodes: u8,
    num_of_ids: u8,

    nodes: [Option<Node>; NUM_OF_NODES],
    ids: [TypeId; NUM_OF_NODES],
}
impl Filter {
    pub fn from<C: Sized + 'static>() -> Self {
        let mut thing = Self {
            num_of_nodes: 1,
            num_of_ids: 1,
            nodes: [None; NUM_OF_NODES],
            ids: [0; NUM_OF_NODES],
        };
        thing.ids[0] = std::intrinsics::type_id::<C>();
        // first node would be the root node
        thing.nodes[0] = Some(Node::Single(false, 0));
        thing
    }

    fn combine(mut first: Self, second: Self, op: Operation) -> Self {
        match op {
            Operation::And => {
                if second == Self::NULL && first == Self::NULL {
                    return Self::NULL;
                } else if second == Self::NULL {
                    return first;
                } else if first == Self::NULL {
                    return second;
                }
            }
            Operation::Or => {
                if second == Self::NULL || first == Self::NULL {
                    return Self::NULL;
                }
            }
            Operation::Xor => {
                if second == Self::NULL && first == Self::NULL {
                    return Self::NULL;
                } else if second == Self::NULL {
                    return !first;
                } else if first == Self::NULL {
                    return !second;
                }
            }
        }

        first.nodes.copy_within(0..first.num_of_nodes as usize, 1);
        first.nodes[0] = None;
        for each in &mut first.nodes[1..(1 + first.num_of_nodes) as usize] {
            match each {
                Some(val) => match val {
                    Node::Single(_, _) => (),
                    Node::Dual(_, index_left, _, index_right) => {
                        match index_left {
                            finally => *finally += 1,
                        }
                        match index_right {
                            finally => *finally += 1,
                        }
                    }
                },
                None => panic!("should not be none"),
            }
        }
        first.num_of_nodes += 1;

        first.nodes
            [first.num_of_nodes as usize..(first.num_of_nodes + second.num_of_nodes) as usize]
            .clone_from_slice(&second.nodes[0..second.num_of_nodes as usize]);
        for each in &mut first.nodes
            [first.num_of_nodes as usize..(first.num_of_nodes + second.num_of_nodes) as usize]
        {
            match each {
                Some(val) => match val {
                    Node::Single(_, _) => (),
                    Node::Dual(_, index_left, _, index_right) => {
                        match index_left {
                            finally => *finally += first.num_of_nodes,
                        }
                        match index_right {
                            finally => *finally += first.num_of_nodes,
                        }
                    }
                },
                None => panic!("should not be none"),
            }
        }
        first.nodes[0] = Some(Node::Dual(false, 1, op, first.num_of_nodes));
        first.num_of_nodes += second.num_of_nodes;

        first.ids
            [first.num_of_ids as usize..first.num_of_ids as usize + second.num_of_ids as usize]
            .clone_from_slice(&second.ids[0..second.num_of_ids as usize]);
        for each in &mut first.nodes
            [(first.num_of_nodes - second.num_of_nodes) as usize..first.num_of_nodes as usize]
        {
            match each {
                Some(val) => match val {
                    Node::Single(_, index) => *index += first.num_of_ids,
                    Node::Dual(_, _, _, _) => (),
                },
                None => panic!("should not be none"),
            }
        }
        first.num_of_ids += second.num_of_ids;

        first
    }

    pub const NULL: Filter = Self {
        num_of_nodes: 0,
        num_of_ids: 0,
        nodes: [None; NUM_OF_NODES],
        ids: [0; NUM_OF_NODES],
    };
}
impl BitAnd for Filter {
    type Output = Filter;
    fn bitand(self, rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::And)
    }
}
impl BitOr for Filter {
    type Output = Filter;
    fn bitor(self, rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::Or)
    }
}
impl BitXor for Filter {
    type Output = Filter;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Filter::combine(self, rhs, Operation::Xor)
    }
}
impl Not for Filter {
    type Output = Filter;
    fn not(mut self) -> Self::Output {
        match &mut self.nodes[0] {
            Some(val) => match val {
                Node::Single(flag, _) => *flag = !*flag,
                Node::Dual(flag, _, _, _) => *flag = !*flag,
            },
            None => panic!("invalid filter"),
        }
        self
    }
}

struct Column {
    ptr: *mut u8,
    cap: usize,
}

impl Column {
    /// it is guranteed that the column would be zeroed and can be diveded by 64
    fn new(size: usize) -> Self {
        assert!(size % 64 == 0);
        let result = Self {
            ptr: unsafe { alloc(Layout::new::<usize>().repeat(size).unwrap().0) },
            cap: size,
        };

        result.as_simd().fill(Simd::splat(0));
        result
    }

    fn double(&mut self) {
        unsafe {
            let ptr = realloc(
                self.ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
                Layout::new::<usize>()
                    .repeat(self.cap * 2)
                    .unwrap()
                    .0
                    .size(),
            );
            if ptr.is_null() {
                panic!("nullptr")
            }
            self.ptr = ptr;
        };

        self.cap *= 2;
        self.as_simd()[self.cap / (64 * 2)..].fill(Simd::splat(0));
    }

    fn as_slice(&self) -> &mut [usize] {
        unsafe { from_raw_parts_mut(self.ptr as *mut usize, self.cap) }
    }

    // this is also very very dangerous
    fn as_simd<'a, 'b>(&'a self) -> &'b mut [Simd<usize, 64>] {
        unsafe { from_raw_parts_mut(self.ptr as *mut Simd<usize, 64>, self.cap / 64) }
    }
}
impl Drop for Column {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, Layout::new::<usize>().repeat(self.cap).unwrap().0) }
    }
}

struct ZST(u8);
const ZST_LAYOUT: Layout = Layout::new::<ZST>();

struct DenseColumn {
    ptr: *mut u8,
    sparse_ptr: *mut u8,
    len: usize,
    cap: usize,
    layout: Layout,
}
impl DenseColumn {
    fn new<C: 'static + Sized>() -> Self {
        let layout = Layout::new::<C>();
        if layout.size() > 0 {
            unsafe {
                Self {
                    ptr: alloc(layout.repeat(4).unwrap().0),
                    sparse_ptr: alloc(Layout::new::<usize>().repeat(4).unwrap().0),
                    len: 0,
                    cap: 4,
                    layout,
                }
            }
        } else {
            unsafe {
                Self {
                    ptr: alloc(ZST_LAYOUT.repeat(4).unwrap().0),
                    sparse_ptr: alloc(Layout::new::<usize>().repeat(4).unwrap().0),
                    len: 0,
                    cap: 4,
                    layout: ZST_LAYOUT,
                }
            }
        }
    }

    // maybe you can box leak this whole thing and reclaim it if doing some change, which also sounds like a massive hack
    fn double(&mut self) {
        unsafe {
            let ptr = realloc(
                self.ptr,
                self.layout.repeat(self.cap).unwrap().0,
                self.layout.repeat(self.cap * 2).unwrap().0.size(),
            );
            if ptr.is_null() {
                panic!("null ptr")
            }
            self.ptr = ptr;

            let ptr_sparse = realloc(
                self.sparse_ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
                Layout::new::<usize>()
                    .repeat(self.cap * 2)
                    .unwrap()
                    .0
                    .size(),
            );
            if ptr_sparse.is_null() {
                panic!("null ptr")
            }
            self.sparse_ptr = ptr_sparse;
        }
        self.cap *= 2;
    }

    /// returns a "naive" dense index, also super dangerous
    fn push<'a, 'b, C: 'static + Sized>(
        &'a mut self,
        value: C,
        sparse_index: SparseIndex,
    ) -> (DenseIndex, &'b mut C) {
        if self.len >= self.cap {
            self.double();
        }
        let dense_index = self.len;
        self.len += 1;
        self.as_sparse_slice()[dense_index] = sparse_index;
        unsafe {
            if self.layout == ZST_LAYOUT {
                copy(&ZST(0u8), &mut self.as_slice::<ZST>()[dense_index], 1);
            } else {
                copy(&value, &mut self.as_slice::<C>()[dense_index], 1);
                forget(value);
            }
        }
        (dense_index, &mut self.as_slice::<C>()[dense_index])
    }

    fn pop<C: 'static + Sized>(&mut self) -> C {
        unsafe {
            let mut value: MaybeUninit<C> = MaybeUninit::uninit();
            if self.layout != ZST_LAYOUT {
                copy::<C>(self.as_slice::<C>().last().unwrap(), value.as_mut_ptr(), 1);
            }
            self.len -= 1;
            value.assume_init()
        }
    }

    // don't think this is zst safe yet
    fn swap_remove<C: 'static + Sized>(&mut self, dense_index: DenseIndex) -> C {
        unsafe {
            let mut value: MaybeUninit<C> = MaybeUninit::uninit();
            if self.layout != ZST_LAYOUT {
                copy::<C>(
                    &mut self.as_slice::<C>()[dense_index],
                    value.as_mut_ptr(),
                    1,
                );
                copy::<C>(
                    self.as_slice::<C>().last().unwrap(),
                    &mut self.as_slice::<C>()[dense_index],
                    1,
                );
            }
            self.as_sparse_slice()[dense_index] = *self.as_sparse_slice().last().unwrap();
            self.len -= 1;
            value.assume_init()
        }
    }

    /// omg this is wildly unsafe, note that slicing into a zst would only yield a slice of 0u8s
    fn as_slice<'a, 'b, C: 'static + Sized>(&'a self) -> &'b mut [C] {
        let layout = Layout::new::<C>();
        if layout == self.layout || layout.size() == 0 {
            unsafe { from_raw_parts_mut(self.ptr as *mut C, self.len) }
        } else {
            panic!("type not matching")
        }
    }

    fn as_sparse_slice(&self) -> &mut [usize] {
        unsafe { from_raw_parts_mut(self.sparse_ptr as *mut usize, self.len) }
    }
}

impl Drop for DenseColumn {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout.repeat(self.cap).unwrap().0);
            dealloc(
                self.sparse_ptr,
                Layout::new::<usize>().repeat(self.cap).unwrap().0,
            );
        }
    }
}

struct State {
    ptr: *mut u8,
    layout: Layout,
}
impl State {
    fn new<C: 'static + Sized>(val: C) -> Self {
        unsafe {
            let layout = Layout::new::<C>();
            let ptr = alloc(layout);
            copy::<C>(&val, ptr as *mut C, 1);
            forget(val);
            Self { ptr, layout }
        }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BufferStatus {
    Vacant,
    Taken,
}

pub struct Table {
    // like what kind of helper????
    helpers: Vec<Column>,
    freehead: SparseIndex,

    current_tick: Wrapping<usize>,

    // helper columns include: sparse index column that tracks which sparse index this row is,
    // status for all the temp buffers
    sparse_index_column: Column,
    // number of components this row has, indexed by sparse indices
    num_of_comp_column: Column,
    // ones and zeros
    zeros_column: Column,
    ones_column: Column,

    // buffer status col,
    buffer_status_column: Vec<BufferStatus>,
    buffers: Vec<Column>,

    table: HashMap<TypeId, (Column, DenseColumn)>,
    states: HashMap<TypeId, State>,
}

impl Table {
    fn new() -> Self {
        let mut result = Self {
            freehead: 1,

            table: HashMap::new(),
            states: HashMap::new(),
            helpers: vec![],
            current_tick: Wrapping(0),

            sparse_index_column: Column::new(64),
            num_of_comp_column: Column::new(64),
            ones_column: Column::new(64),
            zeros_column: Column::new(64),

            buffer_status_column: vec![BufferStatus::Vacant; 8],
            buffers: vec![],
        };

        for _ in 0..10 {
            result.helpers.push(Column::new(64));
        }

        // result.helpers[0].as_simd()[0] = SIMD_START;
        // result.helpers[2].as_slice()[0..5].fill(1);

        result.helpers[4].as_simd().fill(Simd::splat(1));

        // new code
        result.sparse_index_column.as_simd()[0] = SIMD_START;
        result.num_of_comp_column.as_simd()[0] = Simd::splat(0);
        result.ones_column.as_simd()[0] = Simd::splat(1);
        // is this even used????
        result.zeros_column.as_simd()[0] = Simd::splat(0);

        result
    }

    fn cap(&self) -> usize {
        // self.helpers[0].cap
        self.sparse_index_column.cap
    }

    fn double(&mut self) {
        for each in &mut self.helpers {
            each.double();
        }
        for (_, (each, _)) in &mut self.table {
            each.double();
        }

        for each in self.cap() / (2 * 64)..self.cap() / 64 {
            self.sparse_index_column.as_simd()[each] = SIMD_START + Simd::splat(64 * each);
        }
    }

    fn get_free_buffer(&mut self) -> BufferIndex {
        todo!()
    }

    fn apply_operation_then_cache(
        &mut self,
        left: BufferIndex,
        right: BufferIndex,
        invert_flag: bool,
        op: Operation,
    ) -> BufferIndex {
        let left_simd = self.helpers[left].as_simd();
        let right_simd = self.helpers[right].as_simd();
        let mut write_buffer_index = 0;

        // for each in 5..self.helpers.len() {
        //     if self.helpers[2].as_slice()[each] == 0 {
        //         self.helpers[2].as_slice()[each] = 1;
        //         write_buffer_index = each;
        //         break;
        //     }
        // }

        for buffer_index in 0..self.buffers.len() {
            if self.buffer_status_column.as_slice()[buffer_index] == BufferStatus::Vacant {
                self.buffer_status_column[buffer_index] = BufferStatus::Taken;
                write_buffer_index = buffer_index;
                break;
            }
        }

        // this thing actually stretches as you go
        if write_buffer_index == 0 {
            self.helpers.push(Column::new(self.cap()));
            write_buffer_index = self.helpers.len() - 1;
        }

        let write_buffer_simd = self.helpers[write_buffer_index].as_simd();
        match op {
            Operation::And => {
                for each in 0..self.cap() / 64 {
                    write_buffer_simd[each] = left_simd[each] & right_simd[each]
                }
            }
            Operation::Or => {
                for each in 0..self.cap() / 64 {
                    write_buffer_simd[each] = left_simd[each] | right_simd[each]
                }
            }
            Operation::Xor => {
                for each in 0..self.cap() / 64 {
                    write_buffer_simd[each] = left_simd[each] ^ right_simd[each]
                }
            }
        }
        if invert_flag {
            for each in 0..self.cap() / 64 {
                write_buffer_simd[each] = !write_buffer_simd[each];
            }
        }

        // what the fuck does this even do???? just so we don't accidentally free the two preset columns ok
        // make sure these are not preset columns
        // if left != 3 && left != 4 {
        //     // self.helpers[2].as_slice()[left] = 0;
        //     self.buffer_status_column[left] = BufferStatus::Vacant;
        // }
        // if right != 3 && right != 4 {
        //     // self.helpers[2].as_slice()[right] = 0;
        //     self.buffer_status_column[right] = BufferStatus::Vacant;
        // }

        write_buffer_index
    }

    fn filter_traverse(&mut self, node_index: usize, filter: &Filter) -> BufferIndex {
        match &filter.nodes[node_index].unwrap() {
            Node::Single(invert_flag, id_index) => {
                match self.table.get(&filter.ids[*id_index as usize]) {
                    Some((sparse_column, _)) => {
                        let mut buffer_index = 0;
                        for each in 5..self.helpers.len() {
                            if self.helpers[2].as_slice()[each] == 0 {
                                self.helpers[2].as_slice()[each] = 1;
                                buffer_index = each;
                                break;
                            }
                        }
                        if buffer_index == 0 {
                            self.helpers.push(Column::new(self.cap()));
                            buffer_index = self.helpers.len() - 1;
                        }
                        let buffer_simd = self.helpers[buffer_index].as_simd();
                        buffer_simd.clone_from_slice(&sparse_column.as_simd());
                        if *invert_flag {
                            for each in 0..self.cap() / 64 {
                                buffer_simd[each] = !buffer_simd[each];
                            }
                        }
                        buffer_index
                    }
                    None => {
                        if *invert_flag {
                            // this is the ones column
                            4
                        } else {
                            // this is the zeros column
                            3
                        }
                    }
                }
            }

            Node::Dual(invert_flag, node_index_left, op, node_index_right) => {
                let left = self.filter_traverse(*node_index_left as usize, filter);
                let right = self.filter_traverse(*node_index_right as usize, filter);
                // each time you call this only one of the buffers will be freed, after all the branches are
                // resolved only one buffer will remain
                self.apply_operation_then_cache(left, right, *invert_flag, *op)
            }
        }
    }

    fn free_all_buffers(&mut self) {
        // for each in 5..self.helpers.len() {
        //     self.helpers[2].as_slice()[each] = 0;
        // }
        // this is actually a shallow free????
        for each in 5..self.buffer_status_column.len() {
            self.buffer_status_column[each] = BufferStatus::Vacant;
        }
    }

    /// if u want to init a column with nothing in it
    pub fn register_column<C: 'static + Sized>(&mut self) {
        let cap = self.cap();
        self.table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));
    }
    /// the lifetime isn't locked
    pub unsafe fn read_column_raw<'a, 'b, C: 'static + Sized>(
        &'a self,
    ) -> Result<&'b [C], &'static str> {
        match self.table.get(&type_id::<C>()) {
            Some((_, dense_column)) => Ok(dense_column.as_slice()),
            None => Err("type not in table"),
        }
    }

    pub fn insert_new<C: 'static + Sized>(&mut self, value: C) -> SparseIndex {
        let sparse_index = self.freehead;
        for each in sparse_index + 1..self.cap() {
            if self.helpers[1].as_slice()[each] == 0 {
                self.freehead = each;
                break;
            }
        }
        // todo this is SO WRONG wtf
        if sparse_index == self.freehead {
            self.double();
            self.freehead = self.cap() / 2;
        }
        let cap = self.cap();

        let (target_sparse, targe_dense) = self
            .table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));

        let (dense_index, _) = targe_dense.push(value, sparse_index);
        target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;
        targe_dense.as_sparse_slice()[dense_index] = sparse_index;
        self.helpers[1].as_slice()[sparse_index] += 1;
        sparse_index
    }
    pub fn insert_at<C: 'static + Sized>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<SparseIndex, &'static str> {
        let cap = self.cap();
        let (target_sparse, target_dense) = self
            .table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));
        match target_sparse.as_slice()[sparse_index] {
            0 => {
                let (dense_index, _) = target_dense.push(value, sparse_index);
                target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;

                target_dense.as_sparse_slice()[dense_index] = sparse_index;
                self.helpers[1].as_slice()[sparse_index] += 1;
                Ok(sparse_index)
            }
            _ => Err("sparse index at this type column is already taken"),
        }
    }
    pub fn remove<C: 'static + Sized>(
        &mut self,
        sparse_index: SparseIndex,
    ) -> Result<C, &'static str> {
        let (target_sparse, target_dense) = self
            .table
            .get_mut(&type_id::<C>())
            .ok_or("type not in table")?;
        if sparse_index == 0 || sparse_index >= target_sparse.cap {
            return Err("invalid sparse index");
        }

        match target_sparse.as_slice()[sparse_index] {
            0 => Err("sparse index at this type column is empty"),
            _ => {
                let dense_index = target_sparse.as_slice()[sparse_index] & MASK_TAIL;
                target_sparse.as_slice()[sparse_index] = 0;
                self.helpers[1].as_slice()[sparse_index] -= 1;
                if self.helpers[1].as_slice()[sparse_index] == 0 && sparse_index < self.freehead {
                    self.freehead = sparse_index;
                }
                // last element
                let value = if dense_index == target_dense.len - 1 {
                    target_dense.pop()
                } else {
                    let val = target_dense.swap_remove::<C>(dense_index);
                    let sparse_index_to_change = target_dense.as_sparse_slice()[dense_index];
                    target_sparse.as_slice()[sparse_index_to_change] = dense_index | MASK_HEAD;
                    val
                };
                Ok(value)
            }
        }
    }

    // states don't have generations; this is based on hashmap so it actually supports zst states
    pub fn add_state<C: 'static + Sized>(&mut self, res: C) -> Result<(), &'static str> {
        if let Some(_) = self.states.get(&type_id::<C>()) {
            Err("state already present in table")
        } else {
            self.states.insert(type_id::<C>(), State::new::<C>(res));
            Ok(())
        }
    }
    pub fn read_state<C: 'static + Sized>(&mut self) -> Result<&mut C, &'static str> {
        if let Some(state) = self.states.get_mut(&type_id::<C>()) {
            Ok(unsafe { state.ptr.cast::<C>().as_mut().unwrap() })
        } else {
            Err("state never registered")
        }
    }
    pub fn remove_state<C: 'static + Sized>(&mut self) -> Result<C, &'static str> {
        if let Some(state) = self.states.remove(&type_id::<C>()) {
            unsafe {
                let mut value: MaybeUninit<C> = MaybeUninit::uninit();
                copy::<C>(
                    (state.ptr as *mut C).as_mut().unwrap(),
                    value.as_mut_ptr(),
                    1,
                );
                Ok(value.assume_init())
            }
        } else {
            Err("state not registered")
        }
    }

    /// only for IterMut
    unsafe fn read_raw<'a, 'b, C: 'static + Sized>(
        &'a self,
        sparse_index: SparseIndex,
    ) -> &'b mut C {
        let (sparse_column, dense_column) = self.table.get(&type_id::<C>()).unwrap();
        &mut dense_column.as_slice::<C>()[sparse_column.as_slice()[sparse_index] & MASK_TAIL]
    }

    pub fn query_with_filter<C: 'static + Sized>(&mut self, filter: Filter) -> IterMut<C> {
        let filter = Filter::from::<C>() & filter;
        let result_index = self.filter_traverse(0, &filter);
        let result_buffer_simd = self.helpers[result_index].as_simd();
        let sparse_simd = self.sparse_index_column.as_simd();
        for each in 0..self.cap() / 64 {
            result_buffer_simd[each] =
                !((result_buffer_simd[each] >> SHIFT) - ONE) & sparse_simd[each];
        }
        let result_buffer_slice = self.helpers[result_index].as_slice(); // [..largest_occupied_sparse_index]
        result_buffer_slice.sort_unstable_by(|a, b| b.cmp(a));
        IterMut::new(
            result_buffer_slice.as_mut_ptr_range(),
            &self,
            self.current_tick,
        )
    }

    // todo, load/save a single column, then reassemble into a whole table.
    // what if we use a savable trait that has a method to save/load stuff
    pub fn save_column<'a, C: 'static + Sized + Serialize + Deserialize<'a> + Debug>(
        &self,
    ) -> Result<(String, String, String), &'static str> {
        let (sparse_col, dense_col) = self
            .table
            .get(&type_id::<C>())
            .ok_or("no such column in table")?;

        unsafe {
            let temp_sparse_vec =
                Vec::from_raw_parts(sparse_col.ptr as *mut usize, sparse_col.cap, sparse_col.cap);
            let temp_dense_vec =
                Vec::from_raw_parts(dense_col.ptr as *mut C, dense_col.len, dense_col.cap);
            let temp_dense_sparse_vec = Vec::from_raw_parts(
                dense_col.sparse_ptr as *mut usize,
                dense_col.len,
                dense_col.cap,
            );

            let sparse = serde_json::to_string::<Vec<usize>>(&temp_sparse_vec)
                .ok()
                .ok_or("err during serializing sparse")?;
            let dense = serde_json::to_string::<Vec<C>>(&temp_dense_vec)
                .ok()
                .ok_or("err during serializing dense")?;
            let dense_sparse = serde_json::to_string::<Vec<usize>>(&temp_dense_sparse_vec)
                .ok()
                .ok_or("err during serializing dense")?;

            temp_sparse_vec.leak();
            temp_dense_vec.leak();
            temp_dense_sparse_vec.leak();
            Ok((sparse, dense, dense_sparse))
        }
    }

    pub fn load_column<'a, C: 'static + Sized + Serialize + Deserialize<'a> + Debug>(
        &mut self,
        sparse: &'a String,
        dense: &'a String,
        dense_sparse: &'a String,
    ) -> Result<(), &'static str> {
        if self.table.get(&type_id::<C>()).is_some() {
            return Err("type already in table");
        } else {
            let first = serde_json::from_str::<Vec<usize>>(&sparse).unwrap();
            let second = serde_json::from_str::<Vec<C>>(&dense).unwrap();
            let third = serde_json::from_str::<Vec<usize>>(&dense_sparse).unwrap();

            let sparse_col = Column {
                // todo IS THIS EVEN MEMORY SAFE???????
                cap: first.capacity(),
                ptr: first.leak().as_ptr() as _,
            };

            let len = second.len();
            // handle zst
            let dense_col = DenseColumn {
                sparse_ptr: third.leak().as_ptr() as _,
                len,
                cap: second.capacity(),
                layout: if Layout::new::<C>().size() == 0 {
                    ZST_LAYOUT
                } else {
                    Layout::new::<C>()
                },
                ptr: second.leak().as_ptr() as _,
            };

            self.table.insert(type_id::<C>(), (sparse_col, dense_col));

            Ok(())
        }
    }

    // todo, mostly just enumerate all the components and update all component numbers and freehead
    pub fn finalize_loading(&mut self) {
        todo!()
    }
}

// todo consider make this type safer, it currently allows you to save it after adding/removing stuff in
// the respective column, you can also somehow preserve it to the next tick and it still derefs
// so maybe add marker for each tick and after each change is done to the respective column,
// that way you can even make this copy, provided that all changing are from the api that would update these
// indices
pub struct IterMut<'a, C: 'static + Sized> {
    ptr: *mut usize,
    end: *mut usize,

    // todo ticking would clear the buffer, but doing a filtering again would also clear the buffer
    // so instead the buffer should be locked and so should the api for adding/removing components
    tick_index: Wrapping<usize>,
    op_index: Wrapping<usize>,

    table: *const Table,

    _phan: PhantomData<&'a C>,
}
impl<'a, C: 'static + Sized> Debug for IterMut<'a, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice = unsafe { from_raw_parts(self.ptr, self.end.offset_from(self.ptr) as usize) };
        f.debug_struct("IterMutAlt")
            .field("slice: ", &slice)
            .finish()
    }
}
impl<'a, C: 'static + Sized> IterMut<'a, C> {
    fn new(range: Range<*mut usize>, table: &Table, tick_index: Wrapping<usize>) -> Self {
        Self {
            ptr: range.start,
            end: range.end,
            tick_index,
            op_index: Wrapping(0),
            table,
            _phan: PhantomData,
        }
    }
}
impl<'a, C: 'static + Sized> Iterator for IterMut<'a, C> {
    type Item = &'a mut C;
    // type Item = SparseIndex;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.tick_index != (*self.table).current_tick {
                return None;
            }
            if self.ptr != self.end && *self.ptr != 0 {
                let index = *self.ptr;
                self.ptr = self.ptr.add(1);

                let thing = (*self.table).read_raw::<C>(index);
                return Some(thing);
                // return Some(index);
            } else {
                return None;
            }
        }
    }
}

//-----------------ECS-----------------//
pub struct ECS {
    pub table: Table,
    entry_point: fn(&mut Table),
}

impl ECS {
    pub fn new(entry_point: fn(&mut Table)) -> Self {
        Self {
            table: Table::new(),
            entry_point,
        }
    }

    pub fn tick(&mut self) {
        (self.entry_point)(&mut self.table);
        self.table.free_all_buffers();
        self.table.current_tick += 1;
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Serialize, Deserialize, Clone, Copy)]
    struct MyStruct {
        // unfortunately you would have to avoid using &str in any components if you still wanna save them...
        float: f32,
    }

    #[test]
    fn test() {
        // let value = serde_json::to_string_pretty(&[1, 2, 3]).unwrap();
        // println!("{:?}", value)

        let mut table = Table::new();

        for each in 0..200 {
            table.insert_new(MyStruct { float: each as _ });
        }

        let (one, two, three) = table.save_column::<MyStruct>().unwrap();
        // println!("{:?}", two);

        drop(table);

        let mut table = Table::new();

        table.load_column::<MyStruct>(&one, &two, &three).unwrap();
        table.finalize_loading();
        // table.freehead = 200;

        // ok why is this 4096 rather than 1024??
        for each in 200..1000 {
            table.insert_new(MyStruct { float: each as _ });
        }
        println!(
            "{:?}",
            table.table.get(&type_id::<MyStruct>()).unwrap().0.cap
        );

        // todo, there's bug here, it's maybe due to the fact that the column isn't resizing as they should
        println!("{:?}", table.query_with_filter::<MyStruct>(Filter::NULL));

        // for each in table.query_with_filter::<MyStruct>(Filter::NULL) {
        //     print!("{:?}, ", each);
        // }

        // todo, test if column can realloc after getting loaded
    }

    #[test]
    fn foo() {
        let table = Table::new();

        println!("{:?}", table.num_of_comp_column.as_slice());
    }
}
