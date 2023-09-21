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
use std::ops::{BitAnd, BitOr, BitXor, Deref, DerefMut, Index, IndexMut, Not, Range};
use std::ptr::copy;
use std::simd::Simd;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::{alloc::Layout, fmt::Debug};

type SparseIndex = usize;
type DenseIndex = usize;
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
enum BufferIndex {
    ONES,
    ZEROS,
    Index(usize),
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

            None => {
                if self == Self::NULL {
                    self.num_of_nodes = u8::MAX;
                } else {
                    panic!("invalid filter");
                }
            }
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

struct Buffers {
    // ones and zeros
    zeros_column: Column,
    ones_column: Column,

    //kkeep track of the status
    buffer_status_column: Vec<BufferStatus>,

    // actual buffers
    buffers: Vec<Column>,
}

impl Index<BufferIndex> for Buffers {
    type Output = Column;

    fn index(&self, index: BufferIndex) -> &Self::Output {
        match index {
            BufferIndex::ONES => &self.ones_column,
            BufferIndex::ZEROS => &self.zeros_column,
            BufferIndex::Index(val) => &self.buffers[val],
        }
    }
}

impl IndexMut<BufferIndex> for Buffers {
    fn index_mut(&mut self, index: BufferIndex) -> &mut Self::Output {
        match index {
            BufferIndex::ONES => &mut self.ones_column,
            BufferIndex::ZEROS => &mut self.zeros_column,
            BufferIndex::Index(val) => &mut self.buffers[val],
        }
    }
}

impl Buffers {
    fn len(&self) -> usize {
        self.buffers.len()
    }

    fn get_index(&mut self) -> Option<BufferIndex> {
        for index in 0..self.len() {
            if self.buffer_status_column[index] == BufferStatus::Vacant {
                self.buffer_status_column[index] = BufferStatus::Taken;
                return Some(BufferIndex::Index(index));
                break;
            }
        }
        None
    }

    fn new_buffer(&mut self, size: usize) -> BufferIndex {
        self.buffers.push(Column::new(size));
        self.buffer_status_column.push(BufferStatus::Vacant);
        BufferIndex::Index(self.buffers.len() - 1)
    }

    fn free_all(&mut self) {
        // this is actually a shallow free
        self.buffer_status_column.fill(BufferStatus::Vacant);
    }
}

pub struct Table {
    freehead: SparseIndex,

    current_tick: Wrapping<usize>,

    // sparse index column that tracks which sparse index this row is,
    sparse_index_column: Column,
    // number of components this row has, indexed by sparse indices
    num_of_comp_column: Column,

    // all the buffers and buffer related stuff
    buffers: Buffers,

    table: HashMap<TypeId, (Column, DenseColumn)>,
    states: HashMap<TypeId, State>,
}

impl Table {
    fn new() -> Self {
        let mut result = Self {
            freehead: 1,

            table: HashMap::new(),
            states: HashMap::new(),

            current_tick: Wrapping(0),

            sparse_index_column: Column::new(64),
            num_of_comp_column: Column::new(64),

            buffers: Buffers {
                ones_column: Column::new(64),
                zeros_column: Column::new(64),
                buffer_status_column: vec![BufferStatus::Vacant; 8],
                buffers: vec![],
            },
        };

        for _ in 0..8 {
            result.buffers.new_buffer(64);
        }

        result.sparse_index_column.as_simd()[0] = SIMD_START;
        result.num_of_comp_column.as_simd()[0] = Simd::splat(0);
        result.buffers.ones_column.as_simd()[0] = Simd::splat(1);
        result.buffers.zeros_column.as_simd()[0] = Simd::splat(0);

        result
    }

    fn cap(&self) -> usize {
        self.sparse_index_column.cap
    }

    fn double(&mut self) {
        // doubling the sparse col, note that you have to double the sparse col first since we be using
        // self.cap() later on
        self.sparse_index_column.double();

        for each in self.cap() / 128..self.cap() / 64 {
            self.sparse_index_column.as_simd()[each] = SIMD_START + Simd::splat(64 * each);
        }

        // doubling other utility cols
        self.num_of_comp_column.double();
        self.num_of_comp_column.as_simd()[self.cap() / 128..].fill(Simd::splat(0));

        self.buffers.zeros_column.double();
        self.buffers.zeros_column.as_simd()[self.cap() / 128..].fill(Simd::splat(0));

        self.buffers.ones_column.double();
        self.buffers.ones_column.as_simd()[self.cap() / 128..].fill(Simd::splat(1));

        // doubling all the temp buffers
        for each in &mut self.buffers.buffers {
            each.double();
            // buffers don't need to populated i guess
        }

        // doubling all the comp columns
        for (_, (each, _)) in &mut self.table {
            each.double();
        }
    }

    fn apply_operation_then_cache(
        &mut self,
        left: BufferIndex,
        right: BufferIndex,
        invert_flag: bool,
        op: Operation,
    ) -> BufferIndex {
        let left_simd = self.buffers[left].as_simd();
        let right_simd = self.buffers[right].as_simd();

        let mut write_buffer_index = if let Some(val) = self.buffers.get_index() {
            val
        } else {
            self.buffers.new_buffer(self.cap())
        };

        let write_buffer_simd = self.buffers[write_buffer_index].as_simd();

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

        write_buffer_index
    }

    fn filter_traverse(&mut self, node_index: usize, filter: &Filter) -> BufferIndex {
        match &filter.nodes[node_index].unwrap() {
            Node::Single(invert_flag, id_index) => {
                match self.table.get(&filter.ids[*id_index as usize]) {
                    Some((sparse_column, _)) => {
                        let mut buffer_index = if let Some(val) = self.buffers.get_index() {
                            val
                        } else {
                            self.buffers.new_buffer(self.cap())
                        };

                        let buffer_simd = self.buffers[buffer_index].as_simd();

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
                            BufferIndex::ONES
                        } else {
                            // this is the zeros column
                            BufferIndex::ZEROS
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
        self.buffers.free_all();
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
            if self.num_of_comp_column.as_slice()[each] == 0 {
                self.freehead = each;
                break;
            }
        }

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

        self.num_of_comp_column.as_slice()[sparse_index] += 1;

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

                self.num_of_comp_column.as_slice()[sparse_index] += 1;

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

                self.num_of_comp_column.as_slice()[sparse_index] -= 1;
                if self.num_of_comp_column.as_slice()[sparse_index] == 0
                    && sparse_index < self.freehead
                {
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
        if filter == !Filter::NULL {
            return IterMut::new_empty();
        }
        let filter = Filter::from::<C>() & filter;
        // start traversing from root node
        let result_index = self.filter_traverse(0, &filter);
        let result_buffer_simd = self.buffers[result_index].as_simd();
        let sparse_simd = self.sparse_index_column.as_simd();
        for each in 0..self.cap() / 64 {
            result_buffer_simd[each] =
                !((result_buffer_simd[each] >> SHIFT) - ONE) & sparse_simd[each];
        }
        let result_buffer_slice = self.buffers[result_index].as_slice(); // [..largest_occupied_sparse_index]
        result_buffer_slice.sort_unstable_by(|a, b| b.cmp(a));
        IterMut::new(
            result_buffer_slice.as_mut_ptr_range(),
            &self,
            self.current_tick,
        )
    }

    // todo, load/save a single column, then reassemble into a whole table.
    // what if we use a savable trait that has a method to save/load stuff.

    // todo. should lock up the table when the saving just begin, unlock after saving the table metadata
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

            assert!(
                first.capacity() % 64 == 0,
                "column cap isn't a power of 64, possible save data corruption"
            );
            // todo, additional check to make sure that data is not corrupted

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

    pub fn finalize_loading(&mut self) {
        // first it's guaranteed that all columns are power of 64, so alignment is ok
    }
}

// todo intoiter for IterMut

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
        if self.ptr.is_null() {
            f.debug_list().entry(&"empty").finish()
        } else {
            let slice =
                unsafe { from_raw_parts(self.ptr, self.end.offset_from(self.ptr) as usize) };
            f.debug_struct("IterMutAlt")
                .field("slice: ", &slice)
                .finish()
        }
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

    fn new_empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            tick_index: Wrapping(0),
            op_index: Wrapping(0),
            table: std::ptr::null_mut(),
            _phan: PhantomData,
        }
    }
}
impl<'a, C: 'static + Sized> Iterator for IterMut<'a, C> {
    type Item = &'a mut C;
    // type Item = SparseIndex;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.ptr.is_null() {
                return None;
            } else {
                // check if it's still in the same tick between filtering and indexing the result
                if self.tick_index != (*self.table).current_tick {
                    return None;
                }

                // if null then it's an empty self, since index always start from 1 and never 0 we also need to check that???
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
    fn save() {
        let mut table = Table::new();

        // todo ok why is this 4096 rather than 1024??
        for each in 0..200 {
            table.insert_new(MyStruct { float: each as _ });
        }

        let (one, two, three) = table.save_column::<MyStruct>().unwrap();

        drop(table);

        let mut table = Table::new();

        table.load_column::<MyStruct>(&one, &two, &three).unwrap();
        table.finalize_loading();

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
    fn double_test() {
        let mut table = Table::new();

        for _ in 0..2 {
            table.double();
        }

        println!(
            "sparse_index_column: {:?}",
            table.sparse_index_column.as_slice()
        );
        println!(
            "num_of_comp_column: {:?}",
            table.num_of_comp_column.as_slice()
        );

        println!("zeros_column: {:?}", table.buffers.zeros_column.as_slice());
        println!("ones_column: {:?}", table.buffers.ones_column.as_slice());

        for index in 0..8 {
            println!("{:?}", table.buffers.buffers[index].as_slice().len());
        }
    }

    #[test]
    fn insert() {
        let mut table = Table::new();

        for each in 0..200000 {
            table.insert_new::<usize>(each);
        }

        println!("{:?}", table.cap());
    }

    #[test]
    fn filter() {
        let mut table = Table::new();

        for each in 0..200 {
            table.insert_new::<usize>(each);
        }

        for each in 200..2000 {
            table.insert_new::<usize>(each);
        }

        for each in 0..2000 {
            if each % 2 == 0 {
                table.insert_at::<i32>(each, each as i32).unwrap();
            }
        }

        let thing = table.query_with_filter::<usize>(!Filter::NULL);
        println!("{:?}", thing);
    }

    #[test]
    fn iter_empty() {
        let mut thing: IterMut<usize> = IterMut::new_empty();
        println!("{:?}", thing);

        assert!(thing.next().is_none());
    }
}
