/*!
 * HLIR-S: High-Level Intermediate Representation for Synth
 * 
 * This module defines the SSA-based IR with:
 * - Effect tracking (IO, Random, Grad, Quantum, Device)
 * - Linear/affine resource types
 * - Region-based memory management
 * - Shape-aware tensor operations
 */

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

/// Effect types that can be tracked through the program
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Effect {
    /// I/O operations (file, network, console)
    IO,
    /// Random number generation
    Random,
    /// Gradient computation (autodiff)
    Grad,
    /// Quantum operations
    Quantum,
    /// Device operations (GPU, TPU)
    Device(DeviceKind),
    /// User-defined effect
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    CPU,
    GPU(u32), // device index
    TPU(u32),
    Quantum,
    Custom(String),
}

/// Effect set for tracking multiple effects
#[derive(Debug, Clone, Default)]
pub struct EffectSet {
    effects: HashSet<Effect>,
}

impl EffectSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pure() -> Self {
        Self::new()
    }

    pub fn with(effect: Effect) -> Self {
        let mut set = Self::new();
        set.insert(effect);
        set
    }

    pub fn insert(&mut self, effect: Effect) {
        self.effects.insert(effect);
    }

    pub fn union(&mut self, other: &EffectSet) {
        self.effects.extend(other.effects.clone());
    }

    pub fn is_pure(&self) -> bool {
        self.effects.is_empty()
    }

    pub fn has(&self, effect: &Effect) -> bool {
        self.effects.contains(effect)
    }
}

/// Resource ownership model
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceKind {
    /// Can be used exactly once (move semantics)
    Linear,
    /// Can be used at most once
    Affine,
    /// Can be freely copied/shared
    Unrestricted,
}

/// Type representation in HLIR-S
#[derive(Debug, Clone)]
pub enum Type {
    /// Primitive types
    Bool,
    I8, I16, I32, I64, I128,
    U8, U16, U32, U64, U128,
    F16, F32, F64,
    
    /// Tensor with element type and shape
    Tensor(Box<Type>, Shape),
    
    /// Model type with input/output schemas
    Model {
        input: Box<Type>,
        output: Box<Type>,
        resource: ResourceKind,
    },
    
    /// Quantum register
    Qubit(usize), // number of qubits
    
    /// Function type with effects
    Function {
        params: Vec<Type>,
        result: Box<Type>,
        effects: EffectSet,
    },
    
    /// Resource handle
    Resource {
        inner: Box<Type>,
        kind: ResourceKind,
    },
    
    /// Product type (tuple)
    Product(Vec<Type>),
    
    /// Sum type (variant)
    Sum(Vec<(String, Option<Type>)>),
    
    /// Reference (for memory safety)
    Ref(Box<Type>, Mutability),
    
    /// Type variable (for polymorphism)
    Var(TypeVar),
    
    /// Opaque/unknown type
    Unknown,
}

#[derive(Debug, Clone)]
pub enum Mutability {
    Immutable,
    Mutable,
}

#[derive(Debug, Clone)]
pub struct TypeVar {
    pub id: u32,
    pub constraints: Vec<TypeConstraint>,
}

#[derive(Debug, Clone)]
pub enum TypeConstraint {
    Numeric,
    Differentiable,
    Device(DeviceKind),
    Shape(ShapeConstraint),
}

/// Tensor shape representation
#[derive(Debug, Clone)]
pub enum Shape {
    /// Static shape with known dimensions
    Static(Vec<usize>),
    /// Dynamic shape with symbolic dimensions
    Dynamic(Vec<ShapeDim>),
    /// Unknown shape
    Unknown,
}

#[derive(Debug, Clone)]
pub enum ShapeDim {
    /// Known dimension
    Known(usize),
    /// Symbolic dimension (e.g., batch size)
    Symbol(String),
    /// Dimension with constraints
    Constrained {
        symbol: String,
        constraints: Vec<DimConstraint>,
    },
}

#[derive(Debug, Clone)]
pub enum DimConstraint {
    /// Dimension must be divisible by value
    DivisibleBy(usize),
    /// Dimension must be in range
    Range(usize, usize),
    /// Dimension equals another
    Equal(String),
}

#[derive(Debug, Clone)]
pub enum ShapeConstraint {
    /// All dimensions must match
    Equal,
    /// Broadcasting is allowed
    Broadcastable,
    /// Custom constraint
    Custom(String),
}

/// SSA value in HLIR-S
#[derive(Debug, Clone)]
pub struct Value {
    pub id: ValueId,
    pub ty: Type,
    pub def: ValueDef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone)]
pub enum ValueDef {
    /// Function parameter
    Param(u32),
    /// Basic block parameter
    BlockParam(BlockId, u32),
    /// Operation result
    OpResult(OpId, u32),
    /// Constant value
    Const(Constant),
}

#[derive(Debug, Clone)]
pub enum Constant {
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    Tensor(TensorConstant),
}

#[derive(Debug, Clone)]
pub struct TensorConstant {
    pub data: Vec<u8>,
    pub dtype: Type,
    pub shape: Shape,
}

/// Operation in HLIR-S
#[derive(Debug, Clone)]
pub struct Op {
    pub id: OpId,
    pub kind: OpKind,
    pub operands: Vec<ValueId>,
    pub results: Vec<Type>,
    pub effects: EffectSet,
    pub attributes: HashMap<String, Attribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpId(pub u32);

#[derive(Debug, Clone)]
pub enum OpKind {
    // Control flow
    Branch(BlockId),
    CondBranch {
        condition: ValueId,
        true_block: BlockId,
        false_block: BlockId,
    },
    Return(Vec<ValueId>),
    Call {
        func: FuncId,
        args: Vec<ValueId>,
    },
    
    // Tensor operations
    MatMul,
    Conv2D,
    Pool2D(PoolKind),
    Reshape,
    Broadcast,
    Reduce(ReduceOp),
    
    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    
    // AI operations
    ModelLoad(String),
    ModelInfer,
    Prompt(String),
    Embedding,
    Attention,
    
    // Gradient operations
    GradOf(ValueId),
    StopGradient,
    
    // Quantum operations
    QuantumGate(Gate),
    Measure,
    
    // Memory operations
    Alloc(Type, ResourceKind),
    Load,
    Store,
    Free,
    
    // Arithmetic
    Add, Sub, Mul, Div, Rem,
    Neg, Abs,
    
    // Comparison
    Eq, Ne, Lt, Le, Gt, Ge,
    
    // Logical
    And, Or, Not, Xor,
    
    // Probabilistic
    Sample(Distribution),
    Observe,
    LogProb,
    
    // Custom operation
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum PoolKind {
    Max,
    Avg,
    AdaptiveMax,
    AdaptiveAvg,
}

#[derive(Debug, Clone)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

#[derive(Debug, Clone)]
pub enum Gate {
    H,  // Hadamard
    X,  // Pauli-X
    Y,  // Pauli-Y
    Z,  // Pauli-Z
    CNOT,
    Toffoli,
    Phase(f64),
    Rotation(Axis, f64),
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum Axis {
    X, Y, Z,
}

#[derive(Debug, Clone)]
pub enum Distribution {
    Normal { mean: ValueId, std: ValueId },
    Uniform { low: ValueId, high: ValueId },
    Bernoulli { p: ValueId },
    Categorical { probs: ValueId },
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum Attribute {
    Bool(bool),
    I64(i64),
    F64(f64),
    String(String),
    Shape(Shape),
    Type(Type),
    List(Vec<Attribute>),
}

/// Basic block in HLIR-S
#[derive(Debug, Clone)]
pub struct Block {
    pub id: BlockId,
    pub params: Vec<(ValueId, Type)>,
    pub ops: Vec<Op>,
    pub terminator: Op,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// Region (for scoping and lifetime management)
#[derive(Debug, Clone)]
pub struct Region {
    pub id: RegionId,
    pub blocks: Vec<Block>,
    pub entry: BlockId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId(pub u32);

/// Function in HLIR-S
#[derive(Debug, Clone)]
pub struct Function {
    pub id: FuncId,
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub result: Type,
    pub effects: EffectSet,
    pub body: Region,
    pub attributes: HashMap<String, Attribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

/// Module (compilation unit)
#[derive(Debug)]
pub struct Module {
    pub name: String,
    pub functions: HashMap<FuncId, Function>,
    pub globals: HashMap<String, (Type, Option<Constant>)>,
    pub types: HashMap<String, Type>,
    pub effects: HashMap<String, Effect>,
    pub metadata: ModuleMetadata,
}

#[derive(Debug, Default)]
pub struct ModuleMetadata {
    pub version: String,
    pub target: String,
    pub features: HashSet<String>,
    pub dependencies: Vec<String>,
}

/// Builder for constructing HLIR-S
pub struct Builder {
    next_value_id: u32,
    next_op_id: u32,
    next_block_id: u32,
    next_func_id: u32,
    next_region_id: u32,
    current_block: Option<BlockId>,
    blocks: HashMap<BlockId, Block>,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            next_value_id: 0,
            next_op_id: 0,
            next_block_id: 0,
            next_func_id: 0,
            next_region_id: 0,
            current_block: None,
            blocks: HashMap::new(),
        }
    }

    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        self.blocks.insert(id, Block {
            id,
            params: vec![],
            ops: vec![],
            terminator: Op {
                id: OpId(0),
                kind: OpKind::Return(vec![]),
                operands: vec![],
                results: vec![],
                effects: EffectSet::new(),
                attributes: HashMap::new(),
            },
        });
        id
    }

    pub fn set_current_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    pub fn build_op(&mut self, kind: OpKind, operands: Vec<ValueId>, results: Vec<Type>) -> Op {
        let id = OpId(self.next_op_id);
        self.next_op_id += 1;
        
        let effects = self.infer_effects(&kind);
        
        Op {
            id,
            kind,
            operands,
            results,
            effects,
            attributes: HashMap::new(),
        }
    }

    pub fn insert_op(&mut self, op: Op) -> OpId {
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.blocks.get_mut(&block_id) {
                let op_id = op.id;
                block.ops.push(op);
                return op_id;
            }
        }
        panic!("No current block set");
    }

    pub fn create_value(&mut self, ty: Type, def: ValueDef) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    fn infer_effects(&self, kind: &OpKind) -> EffectSet {
        match kind {
            OpKind::ModelLoad(_) | OpKind::ModelInfer => {
                EffectSet::with(Effect::IO)
            }
            OpKind::Sample(_) => {
                EffectSet::with(Effect::Random)
            }
            OpKind::GradOf(_) | OpKind::StopGradient => {
                EffectSet::with(Effect::Grad)
            }
            OpKind::QuantumGate(_) | OpKind::Measure => {
                EffectSet::with(Effect::Quantum)
            }
            OpKind::Alloc(_, _) | OpKind::Free => {
                EffectSet::with(Effect::Device(DeviceKind::CPU))
            }
            _ => EffectSet::new(),
        }
    }
}

/// Verification pass for HLIR-S
pub struct Verifier {
    errors: Vec<String>,
}

impl Verifier {
    pub fn new() -> Self {
        Self { errors: vec![] }
    }

    pub fn verify_module(&mut self, module: &Module) -> Result<(), Vec<String>> {
        for (_, func) in &module.functions {
            self.verify_function(func);
        }
        
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    pub fn verify_function(&mut self, func: &Function) {
        // Verify effect annotations match actual effects
        let actual_effects = self.collect_effects(&func.body);
        for effect in actual_effects.effects {
            if !func.effects.has(&effect) {
                self.errors.push(format!(
                    "Function '{}' has effect {:?} but not declared",
                    func.name, effect
                ));
            }
        }
        
        // Verify resource linearity
        self.verify_resources(&func.body);
        
        // Verify type consistency
        self.verify_types(&func.body);
    }

    fn collect_effects(&self, region: &Region) -> EffectSet {
        let mut effects = EffectSet::new();
        for block in &region.blocks {
            for op in &block.ops {
                effects.union(&op.effects);
            }
            effects.union(&block.terminator.effects);
        }
        effects
    }

    fn verify_resources(&mut self, region: &Region) {
        // Track linear resource usage
        let mut linear_uses: HashMap<ValueId, usize> = HashMap::new();
        
        for block in &region.blocks {
            for op in &block.ops {
                for operand in &op.operands {
                    *linear_uses.entry(*operand).or_insert(0) += 1;
                }
            }
        }
        
        // Check that linear resources are used exactly once
        for (value, uses) in linear_uses {
            if uses != 1 {
                self.errors.push(format!(
                    "Linear resource {:?} used {} times (must be exactly 1)",
                    value, uses
                ));
            }
        }
    }

    fn verify_types(&mut self, region: &Region) {
        // Type checking would go here
        // For now, we'll trust the builder
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::IO => write!(f, "IO"),
            Effect::Random => write!(f, "Random"),
            Effect::Grad => write!(f, "Grad"),
            Effect::Quantum => write!(f, "Quantum"),
            Effect::Device(d) => write!(f, "Device({:?})", d),
            Effect::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pure() {
            write!(f, "pure")
        } else {
            let effects: Vec<String> = self.effects.iter().map(|e| e.to_string()).collect();
            write!(f, "![{}]", effects.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_set() {
        let mut effects = EffectSet::new();
        assert!(effects.is_pure());
        
        effects.insert(Effect::IO);
        assert!(!effects.is_pure());
        assert!(effects.has(&Effect::IO));
        
        let mut effects2 = EffectSet::with(Effect::Random);
        effects2.union(&effects);
        assert!(effects2.has(&Effect::IO));
        assert!(effects2.has(&Effect::Random));
    }

    #[test]
    fn test_builder() {
        let mut builder = Builder::new();
        let block = builder.create_block();
        builder.set_current_block(block);
        
        let v1 = builder.create_value(Type::F32, ValueDef::Const(Constant::F32(1.0)));
        let v2 = builder.create_value(Type::F32, ValueDef::Const(Constant::F32(2.0)));
        
        let add_op = builder.build_op(
            OpKind::Add,
            vec![v1, v2],
            vec![Type::F32],
        );
        
        let op_id = builder.insert_op(add_op);
        assert_eq!(op_id.0, 0);
    }
}