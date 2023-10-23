#![allow(
    dead_code,
    unused_variables,
    unreachable_code,
    unused_mut,
    unused_assignments,
    unused_imports,
    unused_must_use
)]
#![feature(alloc_layout_extra, core_intrinsics, portable_simd, const_type_id)]
#![recursion_limit = "64"]

// todo, consider states also have tags that can be queried and sorted, the reason why i want this is from this scenario
// imagine this is a roguelike, you are chopping a tree with an axe, there is a system that handles the tree and
// the axe components, when and how should we evoke the system, obv by a TREE OF SYSTEMS, then we need some sorta
// data structure the log all the comp locations for sub systems to handle them

// todo, macro

// todo, i feel like having a trait that defines all the api and then have a
// structure that implements all this api is like the better way of doing things??

// todo, thread safety, meaning all that access to anything within table shouldn't cause data race
// preferably using atomics???

// todo, should lock up the table when the saving just begins, unlock after all the stuff is done

// todo, consider change maybeuninit to ptr::read and ptr::write after revamping the zst impl

// todo, so type erased vec, when the data is transferred into the table, what if it's some type of data
// that is thread dependent??? would that still be sound??

// consider also being able to mark entities for deletion, like a tag or a small component if you want additional
// data, soft delete so that other systems trying to work on this entity are less likely to crash, and at the end
// of the ticking, probaly also end of the root system you delete all the components

use arrayvec::ArrayString;
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

const TAG_SIZE: usize = 16;
const NUM_OF_NODES: usize = 16;

const MASK_HEAD: usize = 1 << (usize::BITS - 1);
const MASK_TAIL: usize = !MASK_HEAD;

const SIMD_START: Simd<usize, 64> = Simd::from_array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
]);

const SHIFT: Simd<usize, 64> = Simd::from_array([63; 64]);
const ONE: Simd<usize, 64> = Simd::from_array([1; 64]);

fn process_string(str: &str) -> Result<String, &'static str> {
    if str.is_empty() {
        return Err("empty tag not allowed");
    }
    let string = str.trim().to_lowercase();
    Ok(string)
}

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
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IDorTAG {
    ID(TypeId),
    TAG(ArrayString<TAG_SIZE>),
    None,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Filter {
    num_of_nodes: u8,
    num_of_ids: u8,

    nodes: [Option<Node>; NUM_OF_NODES],
    ids: [IDorTAG; NUM_OF_NODES],
}
impl Filter {
    pub const fn from<C: Sized + 'static>() -> Self {
        let mut thing = Self {
            num_of_nodes: 1,
            num_of_ids: 1,
            nodes: [None; NUM_OF_NODES],
            ids: [IDorTAG::None; NUM_OF_NODES],
        };
        thing.ids[0] = IDorTAG::ID(std::intrinsics::type_id::<C>());
        // first node would be the root node
        thing.nodes[0] = Some(Node::Single(false, 0));
        thing
    }

    pub fn from_tag(tag: &str) -> Self {
        let mut thing = Self {
            num_of_nodes: 1,
            num_of_ids: 1,
            nodes: [None; NUM_OF_NODES],
            ids: [IDorTAG::None; NUM_OF_NODES],
        };
        thing.ids[0] = IDorTAG::TAG(ArrayString::from(tag).unwrap());
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
        ids: [IDorTAG::None; NUM_OF_NODES],
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

#[derive(Serialize, Deserialize)]
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
            // im pretty sure this is relying on undefined behavior to work
            // but then again these would be zst, so no matter the bits are it should
            // be fine
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
    tag_table: HashMap<ArrayString<TAG_SIZE>, Column>,

    states: HashMap<TypeId, State>,
}

impl Table {
    fn new() -> Self {
        let mut result = Self {
            freehead: 1,

            table: HashMap::new(),
            tag_table: HashMap::new(),
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

        let write_buffer_index = if let Some(val) = self.buffers.get_index() {
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
                match &filter.ids[*id_index as usize] {
                    IDorTAG::ID(id) => {
                        match self.table.get(id) {
                            Some((sparse_column, _)) => {
                                let buffer_index = if let Some(val) = self.buffers.get_index() {
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
                    IDorTAG::TAG(tag) => {
                        match self.tag_table.get(tag) {
                            Some(sparse_column) => {
                                let buffer_index = if let Some(val) = self.buffers.get_index() {
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
                    IDorTAG::None => panic!("shouldn't be none at this step"),
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
    // pub unsafe fn read_column_raw<'a, 'b, C: 'static + Sized>(
    //     &'a self,
    // ) -> Result<&'b [C], &'static str> {
    //     match self.table.get(&type_id::<C>()) {
    //         Some((_, dense_column)) => Ok(dense_column.as_slice()),
    //         None => Err("type not in table"),
    //     }
    // }

    pub fn insert<C: 'static + Sized>(&mut self, value: C) -> Access<C> {
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

        let (dense_index, ptr) = targe_dense.push(value, sparse_index);
        target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;
        targe_dense.as_sparse_slice()[dense_index] = sparse_index;

        self.num_of_comp_column.as_slice()[sparse_index] += 1;

        Access::new(ptr, self, sparse_index)
    }

    fn insert_at<C: 'static + Sized>(
        &mut self,
        sparse_index: SparseIndex,
        value: C,
    ) -> Result<Access<C>, &'static str> {
        if sparse_index == 0 {
            return Err("sparse index starts from 1");
        }

        let cap = self.cap();
        let (target_sparse, target_dense) = self
            .table
            .entry(type_id::<C>())
            .or_insert((Column::new(cap), DenseColumn::new::<C>()));
        match target_sparse.as_slice()[sparse_index] {
            0 => {
                let (dense_index, ptr) = target_dense.push(value, sparse_index);
                target_sparse.as_slice()[sparse_index] = dense_index | MASK_HEAD;

                target_dense.as_sparse_slice()[dense_index] = sparse_index;

                self.num_of_comp_column.as_slice()[sparse_index] += 1;

                Ok(Access::new(ptr, self, sparse_index))
            }
            _ => Err("same type as this position is already present"),
        }
    }

    fn remove<C: 'static + Sized>(&mut self, sparse_index: SparseIndex) -> Result<C, &'static str> {
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

                // decreasing the nums by one, if it hits 0 and it left to the current free head it changes the freehead
                // but here is a problem, the access is still valid
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

    // todo consider make the access more thread safe???
    pub fn read_state<'a, 'b, C: 'static + Sized>(&'a mut self) -> Result<&'b mut C, &'static str> {
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

    fn read_direct<C: 'static + Sized>(&mut self, sparse_index: SparseIndex) -> Option<Access<C>> {
        let (sparse_column, dense_column) = self.table.get(&type_id::<C>())?;
        if sparse_index >= self.cap() || sparse_index == 0 {
            return None;
        }
        let mut dense_index = sparse_column.as_slice()[sparse_index];
        if dense_index == 0 {
            return None;
        }
        dense_index &= MASK_TAIL;
        let ptr = &mut dense_column.as_slice::<C>()[dense_index];

        Some(Access::new(ptr, self, sparse_index))
    }

    // todo what if we can only query for a type, but you can filter with tags
    pub fn query<C: 'static + Sized>(&mut self, filter: Filter) -> Result<Query<C>, &'static str> {
        if Layout::new::<C>().size() == 0 {
            return Err("you can't query ZST as target, pls use them as side filters");
        }
        if filter == !Filter::NULL {
            return Ok(Query::new_empty());
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
        Ok(Query::new(result_buffer_slice.as_mut_ptr_range(), self))
    }

    pub fn save_column<'a, C: 'static + Sized + Serialize + Deserialize<'a>>(
        &self,
    ) -> Result<SavedColumn<'a, C>, &'static str> {
        let (sparse_col, dense_col) = self
            .table
            .get(&type_id::<C>())
            .ok_or("no such column in table")?;

        unsafe {
            let temp_sparse_vec =
                Vec::from_raw_parts(sparse_col.ptr as *mut usize, sparse_col.cap, sparse_col.cap);

            let temp_dense_sparse_vec = Vec::from_raw_parts(
                dense_col.sparse_ptr as *mut usize,
                dense_col.len,
                dense_col.cap,
            );

            let sparse = serde_json::to_string::<Vec<usize>>(&temp_sparse_vec)
                .ok()
                .ok_or("err during serializing sparse")?;

            let dense_sparse = serde_json::to_string::<Vec<usize>>(&temp_dense_sparse_vec)
                .ok()
                .ok_or("err during serializing dense")?;

            temp_sparse_vec.leak();
            temp_dense_sparse_vec.leak();

            if Layout::new::<C>().size() == 0 {
                let temp_dense_vec =
                    Vec::from_raw_parts(dense_col.ptr as *mut ZST, dense_col.len, dense_col.cap);
                let dense = serde_json::to_string::<Vec<ZST>>(&temp_dense_vec)
                    .ok()
                    .ok_or("err during serializing dense")?;
                temp_dense_vec.leak();
                return Ok(SavedColumn::new(sparse, dense, dense_sparse));
            } else {
                let temp_dense_vec =
                    Vec::from_raw_parts(dense_col.ptr as *mut C, dense_col.len, dense_col.cap);
                let dense = serde_json::to_string::<Vec<C>>(&temp_dense_vec)
                    .ok()
                    .ok_or("err during serializing dense")?;
                temp_dense_vec.leak();
                return Ok(SavedColumn::new(sparse, dense, dense_sparse));
            };
        }
    }

    pub fn load_column<'a, C: 'static + Sized + Serialize + Deserialize<'a>>(
        &mut self,
        saved_data: &'a SavedColumn<'a, C>,
    ) -> Result<(), &'static str> {
        if self.table.get(&type_id::<C>()).is_some() {
            return Err("type already in table");
        } else {
            let first = serde_json::from_str::<Vec<usize>>(&saved_data.sparse).unwrap();

            let third = serde_json::from_str::<Vec<usize>>(&saved_data.dense_sparse).unwrap();

            assert!(
                first.capacity() != 0
                    && first.capacity() % 64 == 0
                    && ((first.capacity() / 64) & (first.capacity() / 64 - 1)) == 0,
                "column cap isn't a power of 64, possible save data corruption"
            );

            // todo, additional checks to make sure that data is not corrupted,
            // namely the linking between sparse and densesparse
            // and the first bit of all non zero indices
            // note that even so there's no guarantee to whether or
            // not the actual components are corrupted

            let sparse_col = Column {
                cap: first.capacity(),
                ptr: first.leak().as_ptr() as _,
            };

            if Layout::new::<C>().size() == 0 {
                let second = serde_json::from_str::<Vec<ZST>>(&saved_data.dense).unwrap();
                let dense_col = DenseColumn {
                    sparse_ptr: third.leak().as_ptr() as _,
                    len: second.len(),
                    cap: second.capacity(),
                    layout: ZST_LAYOUT,
                    ptr: second.leak().as_ptr() as _,
                };

                self.table.insert(type_id::<C>(), (sparse_col, dense_col));
            } else {
                let second = serde_json::from_str::<Vec<C>>(&saved_data.dense).unwrap();
                let dense_col = DenseColumn {
                    sparse_ptr: third.leak().as_ptr() as _,
                    len: second.len(),
                    cap: second.capacity(),
                    layout: Layout::new::<C>(),
                    ptr: second.leak().as_ptr() as _,
                };

                self.table.insert(type_id::<C>(), (sparse_col, dense_col));
            };

            Ok(())
        }
    }

    pub fn finalize_loading(&mut self) -> Result<(), &'static str> {
        // first it's guaranteed that all columns are power of 64, so alignment is ok
        // and all sparse and dense is linked ok so no worry about that

        // it is also guarateed that there will be free space in all these columns

        // we just need to find the first freehead, and enumerate all num of comp,

        // first we need to make sure all comps are of same size
        let mut table_cap = None::<usize>;
        for (_, (col, _)) in &self.table {
            if table_cap.is_none() {
                table_cap = Some(col.cap);
            } else if table_cap.unwrap() != col.cap {
                return Err("columns aren't of the same size");
            }
        }
        if table_cap.is_none() {
            return Err("there's no comp column in this table");
        }

        // then we need to adjust util cols
        let times_to_double = ((table_cap.unwrap() / 64) as f32).log2() as usize;
        for _ in 0..times_to_double {
            self.sparse_index_column.double();

            for each in self.cap() / 128..self.cap() / 64 {
                self.sparse_index_column.as_simd()[each] = SIMD_START + Simd::splat(64 * each);
            }

            self.num_of_comp_column.double();
            self.num_of_comp_column.as_simd()[self.cap() / 128..].fill(Simd::splat(0));

            self.buffers.zeros_column.double();
            self.buffers.zeros_column.as_simd()[self.cap() / 128..].fill(Simd::splat(0));

            self.buffers.ones_column.double();
            self.buffers.ones_column.as_simd()[self.cap() / 128..].fill(Simd::splat(1));

            for each in &mut self.buffers.buffers {
                each.double();
            }
        }

        let mut first_freehead = None::<SparseIndex>;
        // enumerate all the num of comps
        for index in 1..table_cap.unwrap() {
            let mut temp_counter = 0;
            for (_, (col, _)) in &self.table {
                if col.as_slice()[index] != 0 {
                    temp_counter += 1;
                }
            }
            self.num_of_comp_column.as_slice()[index] = temp_counter;

            if first_freehead.is_none() && temp_counter == 0 {
                first_freehead = Some(index);
            }
        }

        self.freehead = first_freehead.unwrap();

        Ok(())
    }

    fn add_tag(&mut self, str: &str, sparse_index: SparseIndex) -> Result<(), &'static str> {
        let cap = self.cap();
        let col = self
            .tag_table
            .entry(ArrayString::from(str).unwrap())
            .or_insert(Column::new(cap));

        if col.as_slice()[sparse_index] == 0 {
            col.as_slice()[sparse_index] = !0;
            return Ok(());
        } else {
            return Err("tag already in this position");
        }
    }

    fn remove_tag(&mut self, str: &str, sparse_index: SparseIndex) -> Result<(), &'static str> {
        if let Some(col) = self.tag_table.get_mut(str) {
            if col.as_slice()[sparse_index] == 0 {
                Err("tag is in table but not in this position")
            } else {
                col.as_slice()[sparse_index] = 0;
                Ok(())
            }
        } else {
            Err("this tag is not in table")
        }
    }

    fn has_tag(&self, str: &str, sparse_index: SparseIndex) -> bool {
        if let Some(col) = self.tag_table.get(str) {
            if col.as_slice()[sparse_index] == 0 {
                false
            } else {
                true
            }
        } else {
            false
        }
    }
}

pub struct SavedColumn<'a, C: 'static + Sized + Serialize + Deserialize<'a>> {
    pub sparse: String,
    pub dense: String,
    pub dense_sparse: String,

    _phantom: PhantomData<&'a C>,
}

impl<'a, C: 'static + Sized + Serialize + Deserialize<'a>> SavedColumn<'a, C> {
    fn new(sparse: String, dense: String, dense_sparse: String) -> Self {
        Self {
            sparse,
            dense,
            dense_sparse,
            _phantom: PhantomData,
        }
    }
}

// this absolutely does not handle ZST
pub struct Query<C: 'static + Sized> {
    ptr: *mut usize,
    end: *mut usize,

    // todo ticking would clear the buffer, but doing a filtering again would also clear the buffer
    // so instead the buffer should be locked and so should the api for adding/removing components
    tick_index: Wrapping<usize>,
    op_index: Wrapping<usize>,

    table: *mut Table,

    _phan: PhantomData<C>,
}
impl<'a, C: 'static + Sized> Debug for Query<C> {
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
impl<C: 'static + Sized> Query<C> {
    fn new(range: Range<*mut usize>, table: &mut Table) -> Self {
        Self {
            ptr: range.start,
            end: range.end,
            tick_index: table.current_tick,
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
impl<C: 'static + Sized> Iterator for Query<C> {
    type Item = Access<C>;

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

                    (*self.table).read_direct::<C>(index)
                } else {
                    return None;
                }
            }
        }
    }
}

// an access to a row, or an entity, but pinned to a specific component
// rn the access can guarantee that the base comp is always valid
pub struct Access<C: 'static + Sized> {
    ptr: *mut C,
    sparse_index: SparseIndex,
    table: *mut Table,
}

impl<C: 'static + Sized> Access<C> {
    fn new(ptr: &mut C, table: &mut Table, sparse_index: SparseIndex) -> Self {
        Self {
            ptr,
            table,
            sparse_index,
        }
    }

    pub fn access<T: 'static + Sized>(&self) -> Option<Access<T>> {
        unsafe { (*self.table).read_direct::<T>(self.sparse_index) }
    }

    pub fn insert<T: 'static + Sized>(&mut self, val: T) -> Result<Access<T>, &'static str> {
        unsafe { Ok((*(self.table)).insert_at::<T>(self.sparse_index, val)?) }
    }

    pub fn remove<T: 'static + Sized>(&mut self) -> Result<T, &'static str> {
        if type_id::<T>() == type_id::<C>() {
            return Err("cannot remove the base component");
        }
        unsafe { Ok((*self.table).remove::<T>(self.sparse_index)?) }
    }

    pub fn delete(self) -> Result<C, &'static str> {
        unsafe { Ok((*self.table).remove::<C>(self.sparse_index)?) }
    }

    pub fn add_tag(&self, str: &str) -> Result<(), &'static str> {
        unsafe { Ok((*self.table).add_tag(str, self.sparse_index)?) }
    }

    pub fn remove_tag(&mut self, str: &str) -> Result<(), &'static str> {
        unsafe { Ok((*self.table).remove_tag(str, self.sparse_index)?) }
    }

    pub fn has_tag(&self, str: &str) -> bool {
        unsafe { (*self.table).has_tag(str, self.sparse_index) }
    }
}

impl<C: 'static + Sized> Deref for Access<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_mut().unwrap() }
    }
}
impl<C: 'static + Sized> DerefMut for Access<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut().unwrap() }
    }
}

impl<C: 'static + Sized> AsRef<C> for Access<C> {
    fn as_ref(&self) -> &C {
        self.deref()
    }
}

impl<C: 'static + Sized> AsMut<C> for Access<C> {
    fn as_mut(&mut self) -> &mut C {
        self.deref_mut()
    }
}

//-----------------ECS-----------------//
pub struct ECS<T, E> {
    pub table: Table,
    entry_point: fn(&mut Table) -> Result<T, E>,
    error_handle: fn(Result<T, E>),
}

impl<T, E> ECS<T, E> {
    pub fn new(
        entry_point: fn(&mut Table) -> Result<T, E>,
        error_handle: fn(Result<T, E>),
    ) -> Self {
        Self {
            table: Table::new(),
            entry_point,
            error_handle,
        }
    }

    pub fn tick(&mut self) {
        (self.error_handle)((self.entry_point)(&mut self.table));
        self.table.free_all_buffers();
        self.table.current_tick += 1;
    }
}

// ($typ1:tt ^ $typ2:tt) => {
//     type_id::<$typ1>() ^ type_id::<$typ2>()
// };
// ($typ1:tt | $typ2:tt) => {
//     type_id::<$typ1>() | type_id::<$typ2>()
// };
// ($typ1:tt ^ $($typ2:tt)*)=>{
//     type_id::<$typ1>() ^ fil!($($typ2)*)
// };
// ($typ1:tt | $($typ2:tt)*)=>{
//     type_id::<$typ1>() | fil!($($typ2)*)
// };

macro_rules! fil_after {
    // path can be used for traits and structs/enums
    ($type:path) => {
        type_id::<$type>()
    };

    (($($type:tt)*)) => {
        fil_after!($($type)*)
    };

    // recursion stuff
    ($type1:tt & $($type2:tt)* ) => {
        fil_after!($type1) & fil_after!($($type2)*)
    };
    ($type1:tt ^ $($type2:tt)* ) => {
        fil_after!($type1) ^ fil_after!($($type2)*)
    };
    ($type1:tt | $($type2:tt)* ) => {
        fil_after!($type1) | fil_after!($($type2)*)
    };
}

// this makes things respect precedence
macro_rules! fil {
    ($type:ty) => {
        ()
    }; // path can be used for traits and structs/enums
       // ($type:path) => {
       //     $type
       // };

       // (($($type:tt)*)) => {
       //     fil_after!($($type)*)
       // };

       // recursion stuff
       // ($type1:tt & $($type2:tt)* ) => {
       //     fil_after!($type1) & fil_after!($($type2)*)
       // };
       // ($type1:tt ^ $($type2:tt)* ) => {
       //     fil_after!($type1) ^ fil_after!($($type2)*)
       // };
       // ($type1:tt | $($type2:tt)* ) => {
       //     fil_after!($type1) | fil_after!($($type2)*)
       // };
       // ($type1:tt & $type2:tt) => {
       //     fil!($type1) & fil!($type2)
       // };
       // ($type1:tt ^ $type2:tt) => {
       //     fil!($type1) ^ fil!($type2)
       // };
       // ($type1:tt | $type2:tt) => {
       //     fil!($type1) | fil!($type2)
       // };

       // ($($type1:tt)+ & $type2:tt) => {
       //     fil!($type1) & fil!($type2)
       // };
}

fn test_mac_pls() {
    // fil!(&)
    // have to turn this
    // fil!(usize & isize ^ u32) | i32);
    // into this
    // fil_after!(((usize & isize) ^ u32) | i32);

    // have to turn this
    // fil!(usize & isize & u32) & i32);
    // into this
    // fil_after!(((usize & isize) & u32) & i32);

    // have to turn this
    // fil_after!(usize & isize | u32 & i32);
    // into this
    // fil_after!((usize & (isize | u32)) ^ i32);
}

#[cfg(test)]
mod test {
    use std::mem::size_of;

    use super::*;

    #[derive(Serialize, Deserialize, Clone, Copy)]
    struct MyStruct {
        // unfortunately you would have to avoid using &str in any components if you still wanna save them...
        float: f32,
    }

    #[test]
    fn save() {
        let mut table = Table::new();

        for each in 0..200 {
            table.insert(MyStruct { float: each as _ });
        }

        // table.remove::<MyStruct>(120).unwrap();

        for each in 1..101 {
            // table.insert_at::<usize>(each, each as usize).unwrap();
        }

        for each in 150..201 {
            // table.insert_at::<usize>(each, each as usize).unwrap();
        }

        let data_mystruct = table.save_column::<MyStruct>().unwrap();
        let data_usize = table.save_column::<usize>().unwrap();

        drop(table);

        let mut table = Table::new();

        table.load_column::<MyStruct>(&data_mystruct).unwrap();
        table.load_column::<usize>(&data_usize).unwrap();
        table.finalize_loading().unwrap();

        // println!("{:?}", table.num_of_comp_column.as_slice());
        // println!("{:?}", table.freehead);

        for each in 200..1000 {
            table.insert(MyStruct { float: each as _ });
        }
        // println!(
        //     "{:?}",
        //     table
        //         .table
        //         .get(&type_id::<MyStruct>())
        //         .unwrap()
        //         .1
        //         // .as_slice::<MyStruct>()
        //         .as_sparse_slice()
        // );

        // println!(
        //     "{:?}",
        //     table.query_with_filter::<MyStruct>(!Filter::from::<usize>())
        // );
        let mut counter = 0;
        for each in table.query::<MyStruct>(Filter::NULL).unwrap() {
            counter += 1;
        }
        // println!("{:?}", counter);

        // todo valgrind this whole saving thing, im assuming it's not that bad consider it's all using
        // the default allocator
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
            table.insert::<usize>(each);
        }

        println!("{:?}", table.cap());
    }

    #[test]
    fn filter() {
        let mut table = Table::new();

        for each in 0..200 {
            table.insert::<usize>(each);
        }

        for each in 200..2000 {
            table.insert::<usize>(each);
        }

        for each in 0..2000 {
            if each % 2 == 0 {
                // table.insert_at::<i32>(each, each as i32).unwrap();
            }
        }

        let thing = table.query::<usize>(!Filter::NULL);
        println!("{:?}", thing);
    }

    #[test]
    fn iter_empty() {
        let mut thing: Query<usize> = Query::new_empty();
        println!("{:?}", thing);

        assert!(thing.next().is_none());
    }

    #[test]
    fn access() {
        let mut ecs = ECS::new(|_| Ok(()), |_: Result<(), &'static str>| ());

        for each in 1..=20000 {
            ecs.table.insert::<usize>(each);
            // ecs.table.insert_at::<isize>(each, each as isize).unwrap();
            // ecs.table.insert_tag("this").unwrap();
        }
        // println!("{:?}", table.cap());
        // println!("{:?}", table.read_direct::<usize>(0));

        for mut each in ecs.table.query::<usize>(Filter::from::<isize>()).unwrap() {
            *each += 1;
            let mut thing = each.access::<isize>().unwrap();
            // *thing += 1;
            println!("{:?}", *thing);
        }
    }

    #[test]
    fn id() {
        assert!(type_id::<&str>() == type_id::<&str>());
        // for each in query!(table, UwU, fil!(!"enemy" + Health)) {}
        // query!(table, "enemy", fil!(NULL));
        let mut table = Table::new();

        macro_rules! HELLO {
            // ident is var/func name
            // ($thing:ident) => {
            //     println!("hello")
            // };

            // expr is expression and value
            ($table:expr, $type_info:ty /*bitwise operators combination of NULL, types and tags; or leave blank for just NULL*/) => {{
                // println!("{:?} is {:?}", stringify!($thing), $thing);
                // println!("{:?}", type_id::<$another>());
                table.query_with_filter::<$type_info>(Filter::NULL)
            }};

            // AND they don't have to have the same return type
            ($table:expr, $type_info:ty, $filter:tt) => {{
                // println!("{:?} is {:?}", stringify!($thing), $thing);
                // println!("{:?}", type_id::<$another>());
                // table.query_with_filter::<$type_info>(Filter::NULL)
                println!("{:?}", $filter)
            }};
        }

        let thing = HELLO!(table, MyStruct, 12);
        // println!("{:?}", thing);

        macro_rules! filter {
            () => {
                ()
            };
        }

        let foo = filter!();

        macro_rules! recurrence {
            ( a[n]: $sty:ty = $($inits:expr),+; ..., $recur:expr ) => {
                /* ... */
            };
        }

        recurrence![a[n]: u64 = 0, 1;..., a[n-2] + a[n-1]];

        // let thing: ArrayString<128> = ArrayString::from("thiss").unwrap();
    }

    #[derive(PartialEq, Eq, Debug)]
    struct Thing {}
    impl Thing {
        fn hello(&self) {}
    }

    #[test]
    fn zst() {
        let mut table = Table::new();

        for each in 0..10 {
            let mut acc = table.insert(each as usize);
            acc.add_tag("yip").unwrap();
        }

        for mut each in table.query::<usize>(Filter::from_tag("yip")).unwrap() {
            println!("{:?}", *each);
            if each.has_tag("yip") {
                println!("tag is in table uwu");
                if *each % 2 == 0 {
                    each.remove_tag("yip").unwrap();
                }
            }
        }

        // assert!(
        //     table
        //         .query::<usize>(Filter::from_tag("yip"))
        //         .unwrap()
        //         .next()
        //         .is_none()
        //         == true
        // );

        for mut each in table.query::<usize>(!Filter::from_tag("yip")).unwrap() {
            println!("{:?}", *each);
            // if each.has_tag("yip") {
            //     println!("tag is in table uwu");
            //     if *each % 2 == 0 {
            //         each.remove_tag("yip").unwrap();
            //     }
            // }
        }
    }

    #[test]
    fn macr() {
        let i32_id = type_id::<i32>();
        let u32_id = type_id::<u32>();
        let isize_id = type_id::<isize>();
        let usize_id = type_id::<usize>();

        // but they are not
        let one = i32_id & u32_id | isize_id & usize_id;
        let other = i32_id & (u32_id | isize_id) & usize_id;

        assert!(one == other);

        // it would seem that Unique::dangling() is used to handle zst
        let mut thing = Vec::new();
        thing.push(());

        // assert!(i32_id & )
    }

    #[test]
    fn run() {
        let mut table = Table::new();
        let mut access: Vec<Access<Thing>> = vec![];

        for each in 0..10 {
            let mut acc = table.insert(Thing {});
            access.push(acc);
        }

        for mut each in access {
            assert_eq!(each.delete().unwrap(), Thing {});
        }

        let mut thing = vec![(), (), ()];
        let another = thing.as_ptr_range();
        thing.push(());
        assert!(another.end == another.start);

        println!("{:?}", size_of::<Thing>());
    }
}
