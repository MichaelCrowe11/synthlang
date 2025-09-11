/*!
 * Type Checker with Linear/Affine Resources and Effects
 * 
 * This module implements:
 * - Bidirectional type checking
 * - Linear/affine resource tracking
 * - Effect inference and checking
 * - Shape inference for tensors
 * - Automatic differentiation type rules
 */

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::cell::RefCell;

use crate::hlir::{
    Type, EffectSet, Effect, ResourceKind, Shape, ShapeDim, 
    TypeVar, TypeConstraint, Value, ValueId, Op, OpKind,
    Block, Function, Module, DeviceKind, Mutability,
    ShapeConstraint, DimConstraint,
};

/// Type checking context
pub struct TypeContext {
    /// Type environment (variable -> type)
    env: HashMap<ValueId, Type>,
    /// Resource tracking (linear/affine)
    resources: ResourceTracker,
    /// Effect accumulator
    effects: EffectSet,
    /// Type variable substitutions
    subst: HashMap<u32, Type>,
    /// Shape constraints
    shape_constraints: Vec<ShapeConstraint>,
    /// Next type variable ID
    next_tyvar: u32,
    /// Errors collected during checking
    errors: Vec<TypeError>,
}

/// Tracks usage of linear/affine resources
#[derive(Debug, Clone)]
pub struct ResourceTracker {
    /// Linear resources (must be used exactly once)
    linear: HashMap<ValueId, ResourceState>,
    /// Affine resources (can be used at most once)
    affine: HashMap<ValueId, ResourceState>,
    /// Borrowed resources and their lifetimes
    borrows: HashMap<ValueId, BorrowInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceState {
    Available,
    Consumed,
    Borrowed(usize), // borrow count
}

#[derive(Debug, Clone)]
pub struct BorrowInfo {
    pub source: ValueId,
    pub mutability: Mutability,
    pub lifetime: Lifetime,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lifetime {
    pub id: u32,
    pub bounds: Vec<LifetimeBound>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LifetimeBound {
    Outlives(Lifetime),
    Region(u32),
}

#[derive(Debug, Clone)]
pub enum TypeError {
    TypeMismatch {
        expected: Type,
        found: Type,
        location: String,
    },
    LinearResourceError {
        resource: ValueId,
        message: String,
    },
    EffectError {
        effect: Effect,
        message: String,
    },
    ShapeError {
        expected: Shape,
        found: Shape,
        message: String,
    },
    UnificationError {
        ty1: Type,
        ty2: Type,
    },
    UndefinedVariable(ValueId),
    ResourceAlreadyConsumed(ValueId),
    BorrowCheckError(String),
}

impl TypeContext {
    pub fn new() -> Self {
        Self {
            env: HashMap::new(),
            resources: ResourceTracker::new(),
            effects: EffectSet::new(),
            subst: HashMap::new(),
            shape_constraints: Vec::new(),
            next_tyvar: 0,
            errors: Vec::new(),
        }
    }

    /// Create a fresh type variable
    pub fn fresh_tyvar(&mut self) -> Type {
        let id = self.next_tyvar;
        self.next_tyvar += 1;
        Type::Var(TypeVar {
            id,
            constraints: vec![],
        })
    }

    /// Add a type binding
    pub fn bind(&mut self, id: ValueId, ty: Type) {
        // Check for linear/affine resources
        if let Type::Resource { kind, .. } = &ty {
            match kind {
                ResourceKind::Linear => {
                    self.resources.linear.insert(id, ResourceState::Available);
                }
                ResourceKind::Affine => {
                    self.resources.affine.insert(id, ResourceState::Available);
                }
                ResourceKind::Unrestricted => {}
            }
        }
        self.env.insert(id, ty);
    }

    /// Look up a type
    pub fn lookup(&self, id: &ValueId) -> Result<Type, TypeError> {
        self.env.get(id)
            .cloned()
            .ok_or(TypeError::UndefinedVariable(*id))
    }

    /// Use a value (checking resource constraints)
    pub fn use_value(&mut self, id: ValueId) -> Result<Type, TypeError> {
        // Check linear resources
        if let Some(state) = self.resources.linear.get_mut(&id) {
            match state {
                ResourceState::Available => {
                    *state = ResourceState::Consumed;
                }
                ResourceState::Consumed => {
                    return Err(TypeError::ResourceAlreadyConsumed(id));
                }
                ResourceState::Borrowed(_) => {
                    return Err(TypeError::LinearResourceError {
                        resource: id,
                        message: "Cannot use borrowed linear resource".to_string(),
                    });
                }
            }
        }

        // Check affine resources
        if let Some(state) = self.resources.affine.get_mut(&id) {
            match state {
                ResourceState::Available => {
                    *state = ResourceState::Consumed;
                }
                ResourceState::Consumed => {
                    return Err(TypeError::ResourceAlreadyConsumed(id));
                }
                ResourceState::Borrowed(_) => {
                    // Affine resources can be borrowed multiple times
                }
            }
        }

        self.lookup(&id)
    }

    /// Borrow a value
    pub fn borrow_value(&mut self, id: ValueId, mutability: Mutability) -> Result<Type, TypeError> {
        let ty = self.lookup(&id)?;
        
        // Check if we can borrow
        if let Some(state) = self.resources.linear.get_mut(&id) {
            match state {
                ResourceState::Available => {
                    *state = ResourceState::Borrowed(1);
                }
                ResourceState::Consumed => {
                    return Err(TypeError::ResourceAlreadyConsumed(id));
                }
                ResourceState::Borrowed(count) => {
                    if mutability == Mutability::Mutable {
                        return Err(TypeError::BorrowCheckError(
                            "Cannot mutably borrow already borrowed resource".to_string()
                        ));
                    }
                    *state = ResourceState::Borrowed(count + 1);
                }
            }
        }

        Ok(Type::Ref(Box::new(ty), mutability))
    }

    /// Add an effect to the current context
    pub fn add_effect(&mut self, effect: Effect) {
        self.effects.insert(effect);
    }

    /// Get accumulated effects
    pub fn get_effects(&self) -> EffectSet {
        self.effects.clone()
    }

    /// Unify two types
    pub fn unify(&mut self, ty1: &Type, ty2: &Type) -> Result<Type, TypeError> {
        match (ty1, ty2) {
            (Type::Var(v1), Type::Var(v2)) if v1.id == v2.id => Ok(ty1.clone()),
            (Type::Var(v), ty) | (ty, Type::Var(v)) => {
                if let Some(subst_ty) = self.subst.get(&v.id) {
                    self.unify(subst_ty, ty)
                } else {
                    // Check constraints
                    for constraint in &v.constraints {
                        if !self.satisfies_constraint(ty, constraint) {
                            return Err(TypeError::UnificationError {
                                ty1: ty1.clone(),
                                ty2: ty2.clone(),
                            });
                        }
                    }
                    self.subst.insert(v.id, ty.clone());
                    Ok(ty.clone())
                }
            }
            (Type::Tensor(elem1, shape1), Type::Tensor(elem2, shape2)) => {
                let elem = self.unify(elem1, elem2)?;
                let shape = self.unify_shapes(shape1, shape2)?;
                Ok(Type::Tensor(Box::new(elem), shape))
            }
            (Type::Function { params: p1, result: r1, effects: e1 },
             Type::Function { params: p2, result: r2, effects: e2 }) => {
                if p1.len() != p2.len() {
                    return Err(TypeError::TypeMismatch {
                        expected: ty1.clone(),
                        found: ty2.clone(),
                        location: "function arity".to_string(),
                    });
                }
                let mut params = vec![];
                for (t1, t2) in p1.iter().zip(p2.iter()) {
                    params.push(self.unify(t1, t2)?);
                }
                let result = self.unify(r1, r2)?;
                let mut effects = e1.clone();
                effects.union(e2);
                Ok(Type::Function {
                    params,
                    result: Box::new(result),
                    effects,
                })
            }
            _ if self.types_equal(ty1, ty2) => Ok(ty1.clone()),
            _ => Err(TypeError::TypeMismatch {
                expected: ty1.clone(),
                found: ty2.clone(),
                location: "unification".to_string(),
            }),
        }
    }

    /// Check if a type satisfies a constraint
    fn satisfies_constraint(&self, ty: &Type, constraint: &TypeConstraint) -> bool {
        match constraint {
            TypeConstraint::Numeric => matches!(ty, 
                Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::I128 |
                Type::U8 | Type::U16 | Type::U32 | Type::U64 | Type::U128 |
                Type::F16 | Type::F32 | Type::F64
            ),
            TypeConstraint::Differentiable => matches!(ty,
                Type::F16 | Type::F32 | Type::F64 |
                Type::Tensor(elem, _) if matches!(**elem, Type::F16 | Type::F32 | Type::F64)
            ),
            TypeConstraint::Device(device) => {
                // Check if type can be placed on device
                true // For now, assume all types can go on any device
            }
            TypeConstraint::Shape(shape_constraint) => {
                if let Type::Tensor(_, shape) = ty {
                    self.check_shape_constraint(shape, shape_constraint)
                } else {
                    false
                }
            }
        }
    }

    /// Unify two shapes
    fn unify_shapes(&mut self, shape1: &Shape, shape2: &Shape) -> Result<Shape, TypeError> {
        match (shape1, shape2) {
            (Shape::Static(dims1), Shape::Static(dims2)) => {
                if dims1.len() != dims2.len() {
                    return Err(TypeError::ShapeError {
                        expected: shape1.clone(),
                        found: shape2.clone(),
                        message: "Rank mismatch".to_string(),
                    });
                }
                let mut result = vec![];
                for (d1, d2) in dims1.iter().zip(dims2.iter()) {
                    if d1 != d2 && *d1 != 1 && *d2 != 1 {
                        // Broadcasting rules
                        return Err(TypeError::ShapeError {
                            expected: shape1.clone(),
                            found: shape2.clone(),
                            message: format!("Dimension mismatch: {} vs {}", d1, d2),
                        });
                    }
                    result.push((*d1).max(*d2));
                }
                Ok(Shape::Static(result))
            }
            (Shape::Dynamic(dims1), Shape::Dynamic(dims2)) => {
                // More complex dynamic shape unification
                Ok(Shape::Dynamic(dims1.clone())) // Simplified for now
            }
            (Shape::Unknown, shape) | (shape, Shape::Unknown) => Ok(shape.clone()),
            _ => Err(TypeError::ShapeError {
                expected: shape1.clone(),
                found: shape2.clone(),
                message: "Incompatible shape types".to_string(),
            }),
        }
    }

    fn check_shape_constraint(&self, shape: &Shape, constraint: &ShapeConstraint) -> bool {
        match constraint {
            ShapeConstraint::Equal => true, // Would need another shape to compare
            ShapeConstraint::Broadcastable => true, // Always allow broadcasting for now
            ShapeConstraint::Custom(_) => true, // Custom constraints need special handling
        }
    }

    fn types_equal(&self, ty1: &Type, ty2: &Type) -> bool {
        match (ty1, ty2) {
            (Type::Bool, Type::Bool) => true,
            (Type::I8, Type::I8) | (Type::I16, Type::I16) | (Type::I32, Type::I32) |
            (Type::I64, Type::I64) | (Type::I128, Type::I128) => true,
            (Type::U8, Type::U8) | (Type::U16, Type::U16) | (Type::U32, Type::U32) |
            (Type::U64, Type::U64) | (Type::U128, Type::U128) => true,
            (Type::F16, Type::F16) | (Type::F32, Type::F32) | (Type::F64, Type::F64) => true,
            _ => false,
        }
    }
}

impl ResourceTracker {
    pub fn new() -> Self {
        Self {
            linear: HashMap::new(),
            affine: HashMap::new(),
            borrows: HashMap::new(),
        }
    }

    /// Check that all linear resources have been consumed
    pub fn check_linear_consumed(&self) -> Result<(), Vec<TypeError>> {
        let mut errors = vec![];
        for (id, state) in &self.linear {
            if *state == ResourceState::Available {
                errors.push(TypeError::LinearResourceError {
                    resource: *id,
                    message: "Linear resource not consumed".to_string(),
                });
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Type checker for operations
pub struct OpTypeChecker<'a> {
    ctx: &'a mut TypeContext,
}

impl<'a> OpTypeChecker<'a> {
    pub fn new(ctx: &'a mut TypeContext) -> Self {
        Self { ctx }
    }

    pub fn check_op(&mut self, op: &Op) -> Result<Vec<Type>, TypeError> {
        match &op.kind {
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
                self.check_binary_arithmetic(op)
            }
            OpKind::MatMul => self.check_matmul(op),
            OpKind::Conv2D => self.check_conv2d(op),
            OpKind::ReLU | OpKind::Sigmoid | OpKind::Tanh => {
                self.check_activation(op)
            }
            OpKind::Softmax => self.check_softmax(op),
            OpKind::ModelInfer => self.check_model_infer(op),
            OpKind::GradOf(value) => self.check_grad_of(op, *value),
            OpKind::Sample(dist) => self.check_sample(op, dist),
            _ => Ok(vec![Type::Unknown]), // Placeholder for other ops
        }
    }

    fn check_binary_arithmetic(&mut self, op: &Op) -> Result<Vec<Type>, TypeError> {
        if op.operands.len() != 2 {
            return Err(TypeError::TypeMismatch {
                expected: Type::Unknown,
                found: Type::Unknown,
                location: "binary operation arity".to_string(),
            });
        }

        let ty1 = self.ctx.use_value(op.operands[0])?;
        let ty2 = self.ctx.use_value(op.operands[1])?;
        
        let result_ty = self.ctx.unify(&ty1, &ty2)?;
        
        // Check that type is numeric
        let constraint = TypeConstraint::Numeric;
        if !self.ctx.satisfies_constraint(&result_ty, &constraint) {
            return Err(TypeError::TypeMismatch {
                expected: Type::F32, // Example numeric type
                found: result_ty,
                location: "arithmetic operation".to_string(),
            });
        }

        Ok(vec![result_ty])
    }

    fn check_matmul(&mut self, op: &Op) -> Result<Vec<Type>, TypeError> {
        if op.operands.len() != 2 {
            return Err(TypeError::TypeMismatch {
                expected: Type::Unknown,
                found: Type::Unknown,
                location: "matmul arity".to_string(),
            });
        }

        let ty1 = self.ctx.use_value(op.operands[0])?;
        let ty2 = self.ctx.use_value(op.operands[1])?;

        match (&ty1, &ty2) {
            (Type::Tensor(elem1, Shape::Static(dims1)), 
             Type::Tensor(elem2, Shape::Static(dims2))) => {
                // Check element types match
                let elem = self.ctx.unify(elem1, elem2)?;
                
                // Check dimensions are compatible for matmul
                if dims1.len() < 2 || dims2.len() < 2 {
                    return Err(TypeError::ShapeError {
                        expected: Shape::Static(vec![0, 0]),
                        found: Shape::Static(dims1.clone()),
                        message: "MatMul requires at least 2D tensors".to_string(),
                    });
                }

                let m = dims1[dims1.len() - 2];
                let k1 = dims1[dims1.len() - 1];
                let k2 = dims2[dims2.len() - 2];
                let n = dims2[dims2.len() - 1];

                if k1 != k2 {
                    return Err(TypeError::ShapeError {
                        expected: Shape::Static(vec![m, k1]),
                        found: Shape::Static(vec![k2, n]),
                        message: format!("Inner dimensions must match: {} vs {}", k1, k2),
                    });
                }

                // Result shape
                let mut result_dims = dims1[..dims1.len()-2].to_vec();
                result_dims.push(m);
                result_dims.push(n);

                Ok(vec![Type::Tensor(elem, Shape::Static(result_dims))])
            }
            _ => Err(TypeError::TypeMismatch {
                expected: Type::Tensor(Box::new(Type::F32), Shape::Unknown),
                found: ty1,
                location: "matmul".to_string(),
            })
        }
    }

    fn check_conv2d(&mut self, op: &Op) -> Result<Vec<Type>, TypeError> {
        // Simplified conv2d type checking
        if op.operands.len() < 2 {
            return Err(TypeError::TypeMismatch {
                expected: Type::Unknown,
                found: Type::Unknown,
                location: "conv2d arity".to_string(),
            });
        }

        let input = self.ctx.use_value(op.operands[0])?;
        let kernel = self.ctx.use_value(op.operands[1])?;

        match (&input, &kernel) {
            (Type::Tensor(elem1, Shape::Static(dims1)),
             Type::Tensor(elem2, Shape::Static(dims2))) => {
                if dims1.len() != 4 || dims2.len() != 4 {
                    return Err(TypeError::ShapeError {
                        expected: Shape::Static(vec![0, 0, 0, 0]),
                        found: Shape::Static(dims1.clone()),
                        message: "Conv2D expects 4D tensors (NCHW)".to_string(),
                    });
                }

                let elem = self.ctx.unify(elem1, elem2)?;
                
                // Calculate output shape (simplified)
                let batch = dims1[0];
                let out_channels = dims2[0];
                let height = dims1[2]; // Simplified, ignoring padding/stride
                let width = dims1[3];

                Ok(vec![Type::Tensor(elem, Shape::Static(vec![batch, out_channels, height, width]))])
            }
            _ => Err(TypeError::TypeMismatch {
                expected: Type::Tensor(Box::new(Type::F32), Shape::Unknown),
                found: input,
                location: "conv2d".to_string(),
            })
        }
    }

    fn check_activation(&mut self, op: &Op) -> Result<Vec<Type>, TypeError> {
        if op.operands.len() != 1 {
            return Err(TypeError::TypeMismatch {
                expected: Type::Unknown,
                found: Type::Unknown,
                location: "activation arity".to_string(),
            });
        }

        let ty = self.ctx.use_value(op.operands[0])?;
        
        // Activation functions preserve type and shape
        Ok(vec![ty])
    }

    fn check_softmax(&mut self, op: &Op) -> Result<Vec<Type>, TypeError> {
        if op.operands.len() != 1 {
            return Err(TypeError::TypeMismatch {
                expected: Type::Unknown,
                found: Type::Unknown,
                location: "softmax arity".to_string(),
            });
        }

        let ty = self.ctx.use_value(op.operands[0])?;
        
        // Softmax preserves shape but ensures float type
        match ty {
            Type::Tensor(elem, shape) => {
                let float_elem = match *elem {
                    Type::F16 | Type::F32 | Type::F64 => *elem,
                    _ => Type::F32, // Default to F32
                };
                Ok(vec![Type::Tensor(Box::new(float_elem), shape)])
            }
            _ => Err(TypeError::TypeMismatch {
                expected: Type::Tensor(Box::new(Type::F32), Shape::Unknown),
                found: ty,
                location: "softmax".to_string(),
            })
        }
    }

    fn check_model_infer(&mut self, op: &Op) -> Result<Vec<Type>, TypeError> {
        if op.operands.len() < 2 {
            return Err(TypeError::TypeMismatch {
                expected: Type::Unknown,
                found: Type::Unknown,
                location: "model_infer arity".to_string(),
            });
        }

        let model_ty = self.ctx.use_value(op.operands[0])?;
        let input_ty = self.ctx.use_value(op.operands[1])?;

        // Add IO effect
        self.ctx.add_effect(Effect::IO);

        match model_ty {
            Type::Model { input, output, .. } => {
                // Check input type matches
                self.ctx.unify(&*input, &input_ty)?;
                Ok(vec![*output])
            }
            _ => Err(TypeError::TypeMismatch {
                expected: Type::Model {
                    input: Box::new(Type::Unknown),
                    output: Box::new(Type::Unknown),
                    resource: ResourceKind::Linear,
                },
                found: model_ty,
                location: "model_infer".to_string(),
            })
        }
    }

    fn check_grad_of(&mut self, op: &Op, value: ValueId) -> Result<Vec<Type>, TypeError> {
        let ty = self.ctx.lookup(&value)?;
        
        // Add Grad effect
        self.ctx.add_effect(Effect::Grad);
        
        // Check type is differentiable
        let constraint = TypeConstraint::Differentiable;
        if !self.ctx.satisfies_constraint(&ty, &constraint) {
            return Err(TypeError::TypeMismatch {
                expected: Type::F32,
                found: ty.clone(),
                location: "grad_of: type must be differentiable".to_string(),
            });
        }

        // Gradient has same type as input
        Ok(vec![ty])
    }

    fn check_sample(&mut self, op: &Op, dist: &crate::hlir::Distribution) -> Result<Vec<Type>, TypeError> {
        // Add Random effect
        self.ctx.add_effect(Effect::Random);
        
        // Determine sample type based on distribution
        let sample_ty = match dist {
            crate::hlir::Distribution::Normal { .. } |
            crate::hlir::Distribution::Uniform { .. } => Type::F32,
            crate::hlir::Distribution::Bernoulli { .. } => Type::Bool,
            crate::hlir::Distribution::Categorical { probs } => {
                // Check probs type
                let probs_ty = self.ctx.use_value(*probs)?;
                match probs_ty {
                    Type::Tensor(_, Shape::Static(dims)) if dims.len() == 1 => Type::I32,
                    _ => return Err(TypeError::TypeMismatch {
                        expected: Type::Tensor(Box::new(Type::F32), Shape::Static(vec![0])),
                        found: probs_ty,
                        location: "categorical distribution".to_string(),
                    })
                }
            }
            crate::hlir::Distribution::Custom(_) => Type::Unknown,
        };

        Ok(vec![sample_ty])
    }
}

/// Type check a function
pub fn type_check_function(func: &Function) -> Result<(), Vec<TypeError>> {
    let mut ctx = TypeContext::new();
    
    // Add parameters to context
    for (i, (name, ty)) in func.params.iter().enumerate() {
        ctx.bind(ValueId(i as u32), ty.clone());
    }

    // Type check function body
    for block in &func.body.blocks {
        type_check_block(&mut ctx, block)?;
    }

    // Check that declared effects match actual effects
    let actual_effects = ctx.get_effects();
    for effect in actual_effects.effects {
        if !func.effects.has(&effect) {
            ctx.errors.push(TypeError::EffectError {
                effect,
                message: format!("Effect not declared in function signature"),
            });
        }
    }

    // Check that all linear resources are consumed
    ctx.resources.check_linear_consumed()?;

    if ctx.errors.is_empty() {
        Ok(())
    } else {
        Err(ctx.errors)
    }
}

/// Type check a block
pub fn type_check_block(ctx: &mut TypeContext, block: &Block) -> Result<(), Vec<TypeError>> {
    // Add block parameters
    for (id, ty) in &block.params {
        ctx.bind(*id, ty.clone());
    }

    // Type check operations
    let mut checker = OpTypeChecker::new(ctx);
    for op in &block.ops {
        let result_types = checker.check_op(op).map_err(|e| vec![e])?;
        
        // Bind results
        for (i, ty) in result_types.iter().enumerate() {
            let result_id = ValueId(op.id.0 * 100 + i as u32); // Simple ID scheme
            ctx.bind(result_id, ty.clone());
        }
    }

    // Type check terminator
    checker.check_op(&block.terminator).map_err(|e| vec![e])?;

    Ok(())
}

/// Type check a module
pub fn type_check_module(module: &Module) -> Result<(), Vec<TypeError>> {
    let mut all_errors = vec![];
    
    for (_, func) in &module.functions {
        if let Err(errors) = type_check_function(func) {
            all_errors.extend(errors);
        }
    }

    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(all_errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_context() {
        let mut ctx = TypeContext::new();
        let id = ValueId(0);
        ctx.bind(id, Type::F32);
        
        let ty = ctx.lookup(&id).unwrap();
        assert!(matches!(ty, Type::F32));
    }

    #[test]
    fn test_linear_resource() {
        let mut ctx = TypeContext::new();
        let id = ValueId(0);
        ctx.bind(id, Type::Resource {
            inner: Box::new(Type::F32),
            kind: ResourceKind::Linear,
        });

        // First use should succeed
        assert!(ctx.use_value(id).is_ok());
        
        // Second use should fail
        assert!(ctx.use_value(id).is_err());
    }

    #[test]
    fn test_unification() {
        let mut ctx = TypeContext::new();
        
        let ty1 = Type::Tensor(Box::new(Type::F32), Shape::Static(vec![10, 20]));
        let ty2 = Type::Tensor(Box::new(Type::F32), Shape::Static(vec![10, 20]));
        
        let unified = ctx.unify(&ty1, &ty2).unwrap();
        assert!(matches!(unified, Type::Tensor(..)));
    }

    #[test]
    fn test_effect_tracking() {
        let mut ctx = TypeContext::new();
        
        assert!(ctx.get_effects().is_pure());
        
        ctx.add_effect(Effect::IO);
        ctx.add_effect(Effect::Random);
        
        let effects = ctx.get_effects();
        assert!(effects.has(&Effect::IO));
        assert!(effects.has(&Effect::Random));
    }
}