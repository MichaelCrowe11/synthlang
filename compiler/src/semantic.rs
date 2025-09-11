/*!
 * SYNTH Semantic Analysis
 * Type checking, name resolution, and semantic validation
 */

use std::collections::HashMap;
use crate::ast::{self, Node, Expression, Statement, Declaration, Program};
use crate::types::{Type, TypeEnvironment, TypeInference, FunctionType, Effect};
use crate::diagnostics::{DiagnosticEngine, Diagnostic, Span};

pub struct SemanticAnalyzer {
    env: TypeEnvironment,
    diagnostics: DiagnosticEngine,
    current_loop_depth: usize,
    current_function: Option<String>,
    imported_modules: HashMap<String, Vec<String>>,
}

impl SemanticAnalyzer {
    pub fn new(diagnostics: &mut DiagnosticEngine) -> Self {
        Self {
            env: TypeEnvironment::new(),
            diagnostics: DiagnosticEngine::new(),
            current_loop_depth: 0,
            current_function: None,
            imported_modules: HashMap::new(),
        }
    }
    
    pub fn analyze(&mut self, program: &mut Program) -> Result<(), Vec<Diagnostic>> {
        // First pass: collect type definitions and function signatures
        self.collect_declarations(program)?;
        
        // Second pass: type check function bodies and expressions
        self.check_program(program)?;
        
        // Third pass: verify semantic constraints
        self.verify_constraints(program)?;
        
        if self.diagnostics.has_errors() {
            Err(self.diagnostics.take_diagnostics())
        } else {
            Ok(())
        }
    }
    
    fn collect_declarations(&mut self, program: &Program) -> Result<(), ()> {
        for item in &program.items {
            match item {
                Declaration::Function(func) => {
                    let func_type = self.function_signature_to_type(func);
                    self.env.define(func.name.value.clone(), Type::Function(func_type));
                }
                
                Declaration::Struct(struct_decl) => {
                    // Register struct type
                    let struct_type = self.struct_to_type(struct_decl);
                    self.env.define_type(&struct_decl.name.value, 
                        crate::types::TypeDefinition::Struct(struct_type));
                }
                
                Declaration::Enum(enum_decl) => {
                    // Register enum type
                    let enum_type = self.enum_to_type(enum_decl);
                    self.env.define_type(&enum_decl.name.value,
                        crate::types::TypeDefinition::Enum(enum_type));
                }
                
                Declaration::Interface(interface_decl) => {
                    // Register interface type
                    let interface_type = self.interface_to_type(interface_decl);
                    self.env.define_type(&interface_decl.name.value,
                        crate::types::TypeDefinition::Interface(interface_type));
                }
                
                _ => {}
            }
        }
        Ok(())
    }
    
    fn check_program(&mut self, program: &mut Program) -> Result<(), ()> {
        for item in &mut program.items {
            self.check_declaration(item)?;
        }
        Ok(())
    }
    
    fn check_declaration(&mut self, decl: &mut Declaration) -> Result<(), ()> {
        match decl {
            Declaration::Function(func) => {
                self.check_function(func)?;
            }
            
            Declaration::Variable(var) => {
                let init_type = self.infer_expression(&var.initializer)?;
                
                if let Some(ref type_annotation) = var.type_annotation {
                    let declared_type = self.ast_type_to_type(type_annotation);
                    
                    if !init_type.is_compatible_with(&declared_type) {
                        self.diagnostics.error(format!(
                            "Type mismatch: expected {}, found {}",
                            declared_type, init_type
                        ))
                        .with_span(0, var.name.span)
                        .emit();
                        return Err(());
                    }
                    
                    self.env.define(var.name.value.clone(), declared_type);
                } else {
                    self.env.define(var.name.value.clone(), init_type);
                }
            }
            
            Declaration::Constant(const_decl) => {
                let init_type = self.infer_expression(&const_decl.initializer)?;
                
                if let Some(ref type_annotation) = const_decl.type_annotation {
                    let declared_type = self.ast_type_to_type(type_annotation);
                    
                    if !init_type.is_compatible_with(&declared_type) {
                        self.diagnostics.error(format!(
                            "Type mismatch in constant: expected {}, found {}",
                            declared_type, init_type
                        ))
                        .with_span(0, const_decl.name.span)
                        .emit();
                        return Err(());
                    }
                    
                    self.env.define(const_decl.name.value.clone(), declared_type);
                } else {
                    self.env.define(const_decl.name.value.clone(), init_type);
                }
            }
            
            _ => {}
        }
        Ok(())
    }
    
    fn check_function(&mut self, func: &ast::Function) -> Result<(), ()> {
        self.env.push_scope();
        self.current_function = Some(func.name.value.clone());
        
        // Add parameters to scope
        for param in &func.parameters {
            let param_type = self.ast_type_to_type(&param.type_annotation);
            self.env.define(param.name.value.clone(), param_type);
        }
        
        // Set current function type for return type checking
        let func_type = self.function_signature_to_type(func);
        self.env.set_current_function(func_type);
        
        // Check function body
        let body_type = self.check_block(&func.body)?;
        
        // Verify return type
        if let Some(ref return_type) = func.return_type {
            let expected = self.ast_type_to_type(return_type);
            
            if !body_type.is_compatible_with(&expected) {
                self.diagnostics.error(format!(
                    "Function '{}' return type mismatch: expected {}, found {}",
                    func.name.value, expected, body_type
                ))
                .with_span(0, func.name.span)
                .emit();
                return Err(());
            }
        }
        
        self.env.clear_current_function();
        self.current_function = None;
        self.env.pop_scope();
        
        Ok(())
    }
    
    fn check_block(&mut self, block: &ast::Block) -> Result<Type, ()> {
        let mut last_type = Type::Unit;
        
        for stmt in &block.statements {
            last_type = self.check_statement(stmt)?;
        }
        
        Ok(last_type)
    }
    
    fn check_statement(&mut self, stmt: &Statement) -> Result<Type, ()> {
        match stmt {
            Statement::Expression(expr) => self.infer_expression(expr),
            
            Statement::Return(expr) => {
                let expr_type = if let Some(e) = expr {
                    self.infer_expression(e)?
                } else {
                    Type::Unit
                };
                
                if let Some(expected) = self.env.current_return_type() {
                    if !expr_type.is_compatible_with(expected) {
                        self.diagnostics.error(format!(
                            "Return type mismatch: expected {}, found {}",
                            expected, expr_type
                        ))
                        .emit();
                        return Err(());
                    }
                }
                
                Ok(Type::Never)
            }
            
            Statement::If { condition, then_block, else_block } => {
                let cond_type = self.infer_expression(condition)?;
                
                if !cond_type.is_compatible_with(&Type::Bool) {
                    self.diagnostics.error(format!(
                        "If condition must be boolean, found {}",
                        cond_type
                    ))
                    .emit();
                    return Err(());
                }
                
                self.env.push_scope();
                let then_type = self.check_block(then_block)?;
                self.env.pop_scope();
                
                let else_type = if let Some(else_b) = else_block {
                    self.env.push_scope();
                    let t = self.check_block(else_b)?;
                    self.env.pop_scope();
                    t
                } else {
                    Type::Unit
                };
                
                // Unify branch types
                then_type.unify(&else_type).ok_or_else(|| {
                    self.diagnostics.error(format!(
                        "If branches have incompatible types: {} and {}",
                        then_type, else_type
                    ))
                    .emit();
                })
            }
            
            Statement::While { condition, body } => {
                let cond_type = self.infer_expression(condition)?;
                
                if !cond_type.is_compatible_with(&Type::Bool) {
                    self.diagnostics.error(format!(
                        "While condition must be boolean, found {}",
                        cond_type
                    ))
                    .emit();
                    return Err(());
                }
                
                self.current_loop_depth += 1;
                self.env.push_scope();
                self.check_block(body)?;
                self.env.pop_scope();
                self.current_loop_depth -= 1;
                
                Ok(Type::Unit)
            }
            
            Statement::For { variable, iterable, body } => {
                let iter_type = self.infer_expression(iterable)?;
                
                // Determine element type
                let elem_type = match &iter_type {
                    Type::Array(t) => (**t).clone(),
                    Type::String => Type::String,
                    _ => {
                        self.diagnostics.error(format!(
                            "Cannot iterate over type {}",
                            iter_type
                        ))
                        .emit();
                        return Err(());
                    }
                };
                
                self.current_loop_depth += 1;
                self.env.push_scope();
                self.env.define(variable.value.clone(), elem_type);
                self.check_block(body)?;
                self.env.pop_scope();
                self.current_loop_depth -= 1;
                
                Ok(Type::Unit)
            }
            
            Statement::Break => {
                if self.current_loop_depth == 0 {
                    self.diagnostics.error("Break outside of loop")
                        .emit();
                    return Err(());
                }
                Ok(Type::Never)
            }
            
            Statement::Continue => {
                if self.current_loop_depth == 0 {
                    self.diagnostics.error("Continue outside of loop")
                        .emit();
                    return Err(());
                }
                Ok(Type::Never)
            }
            
            _ => Ok(Type::Unit),
        }
    }
    
    fn infer_expression(&mut self, expr: &Node<Expression>) -> Result<Type, ()> {
        match &expr.value {
            Expression::Literal(lit) => Ok(self.infer_literal(lit)),
            
            Expression::Identifier(name) => {
                self.env.lookup(name).cloned().ok_or_else(|| {
                    self.diagnostics.error(format!("Undefined variable '{}'", name))
                        .with_span(0, expr.span)
                        .emit();
                })
            }
            
            Expression::Binary { op, left, right } => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;
                
                self.infer_binary_op(op, &left_type, &right_type, expr.span)
            }
            
            Expression::Unary { op, operand } => {
                let operand_type = self.infer_expression(operand)?;
                self.infer_unary_op(op, &operand_type, expr.span)
            }
            
            Expression::Call { function, arguments } => {
                let func_type = self.infer_expression(function)?;
                
                match func_type {
                    Type::Function(ft) => {
                        if arguments.len() != ft.params.len() {
                            self.diagnostics.error(format!(
                                "Function expects {} arguments, got {}",
                                ft.params.len(), arguments.len()
                            ))
                            .with_span(0, expr.span)
                            .emit();
                            return Err(());
                        }
                        
                        for (arg, param_type) in arguments.iter().zip(&ft.params) {
                            let arg_type = self.infer_expression(arg)?;
                            if !arg_type.is_compatible_with(param_type) {
                                self.diagnostics.error(format!(
                                    "Argument type mismatch: expected {}, found {}",
                                    param_type, arg_type
                                ))
                                .with_span(0, arg.span)
                                .emit();
                                return Err(());
                            }
                        }
                        
                        Ok(*ft.return_type)
                    }
                    _ => {
                        self.diagnostics.error(format!(
                            "Cannot call non-function type {}",
                            func_type
                        ))
                        .with_span(0, expr.span)
                        .emit();
                        Err(())
                    }
                }
            }
            
            Expression::Array(elements) => {
                if elements.is_empty() {
                    Ok(Type::Array(Box::new(Type::Unknown)))
                } else {
                    let first_type = self.infer_expression(&elements[0])?;
                    
                    for elem in &elements[1..] {
                        let elem_type = self.infer_expression(elem)?;
                        if !elem_type.is_compatible_with(&first_type) {
                            self.diagnostics.error(format!(
                                "Array elements must have consistent types: expected {}, found {}",
                                first_type, elem_type
                            ))
                            .with_span(0, elem.span)
                            .emit();
                            return Err(());
                        }
                    }
                    
                    Ok(Type::Array(Box::new(first_type)))
                }
            }
            
            Expression::Index { object, index } => {
                let obj_type = self.infer_expression(object)?;
                let idx_type = self.infer_expression(index)?;
                
                match obj_type {
                    Type::Array(elem_type) => {
                        if !idx_type.is_compatible_with(&Type::Int) {
                            self.diagnostics.error(format!(
                                "Array index must be integer, found {}",
                                idx_type
                            ))
                            .with_span(0, index.span)
                            .emit();
                            return Err(());
                        }
                        Ok(*elem_type)
                    }
                    Type::String => {
                        if !idx_type.is_compatible_with(&Type::Int) {
                            self.diagnostics.error(format!(
                                "String index must be integer, found {}",
                                idx_type
                            ))
                            .with_span(0, index.span)
                            .emit();
                            return Err(());
                        }
                        Ok(Type::String)
                    }
                    _ => {
                        self.diagnostics.error(format!(
                            "Cannot index type {}",
                            obj_type
                        ))
                        .with_span(0, object.span)
                        .emit();
                        Err(())
                    }
                }
            }
            
            Expression::MemberAccess { object, member } => {
                let obj_type = self.infer_expression(object)?;
                self.infer_member_access(&obj_type, &member.value, expr.span)
            }
            
            Expression::AiGenerate { prompt, .. } => {
                let prompt_type = self.infer_expression(prompt)?;
                
                if !prompt_type.is_compatible_with(&Type::String) {
                    self.diagnostics.error(format!(
                        "AI prompt must be string, found {}",
                        prompt_type
                    ))
                    .with_span(0, prompt.span)
                    .emit();
                    return Err(());
                }
                
                Ok(Type::String)
            }
            
            Expression::Embed(text) => {
                let text_type = self.infer_expression(text)?;
                
                if !text_type.is_compatible_with(&Type::String) {
                    self.diagnostics.error(format!(
                        "Embed input must be string, found {}",
                        text_type
                    ))
                    .with_span(0, text.span)
                    .emit();
                    return Err(());
                }
                
                Ok(Type::Embedding(1536)) // Default embedding dimension
            }
            
            Expression::SemanticSimilarity { left, right } => {
                let left_type = self.infer_expression(left)?;
                let right_type = self.infer_expression(right)?;
                
                if !left_type.supports_ai_ops() || !right_type.supports_ai_ops() {
                    self.diagnostics.error(format!(
                        "Semantic similarity requires AI-compatible types, found {} and {}",
                        left_type, right_type
                    ))
                    .with_span(0, expr.span)
                    .emit();
                    return Err(());
                }
                
                Ok(Type::Float)
            }
            
            Expression::Uncertainty { value, confidence } => {
                let value_type = self.infer_expression(value)?;
                let conf_type = self.infer_expression(confidence)?;
                
                if !conf_type.is_compatible_with(&Type::Float) {
                    self.diagnostics.error(format!(
                        "Confidence must be float, found {}",
                        conf_type
                    ))
                    .with_span(0, confidence.span)
                    .emit();
                    return Err(());
                }
                
                Ok(Type::Uncertain(Box::new(value_type)))
            }
            
            _ => Ok(Type::Unknown),
        }
    }
    
    fn infer_literal(&self, lit: &ast::Literal) -> Type {
        match lit {
            ast::Literal::Integer(_) => Type::Int,
            ast::Literal::Float(_) => Type::Float,
            ast::Literal::String(_) => Type::String,
            ast::Literal::Boolean(_) => Type::Bool,
            ast::Literal::Null => Type::Optional(Box::new(Type::Unknown)),
        }
    }
    
    fn infer_binary_op(&mut self, op: &ast::BinaryOperator, left: &Type, right: &Type, span: Span) -> Result<Type, ()> {
        use ast::BinaryOperator::*;
        
        match op {
            Add | Subtract | Multiply | Divide | Modulo => {
                if !left.is_numeric() || !right.is_numeric() {
                    self.diagnostics.error(format!(
                        "Arithmetic operations require numeric types, found {} and {}",
                        left, right
                    ))
                    .with_span(0, span)
                    .emit();
                    return Err(());
                }
                
                left.unify(right).ok_or_else(|| {
                    self.diagnostics.error(format!(
                        "Cannot unify types {} and {} in arithmetic operation",
                        left, right
                    ))
                    .with_span(0, span)
                    .emit();
                })
            }
            
            Equal | NotEqual => {
                if !left.is_comparable() || !right.is_comparable() {
                    self.diagnostics.error(format!(
                        "Comparison requires comparable types, found {} and {}",
                        left, right
                    ))
                    .with_span(0, span)
                    .emit();
                    return Err(());
                }
                Ok(Type::Bool)
            }
            
            Less | LessEqual | Greater | GreaterEqual => {
                if !left.is_numeric() || !right.is_numeric() {
                    self.diagnostics.error(format!(
                        "Ordering comparison requires numeric types, found {} and {}",
                        left, right
                    ))
                    .with_span(0, span)
                    .emit();
                    return Err(());
                }
                Ok(Type::Bool)
            }
            
            And | Or => {
                if !left.is_compatible_with(&Type::Bool) || !right.is_compatible_with(&Type::Bool) {
                    self.diagnostics.error(format!(
                        "Logical operations require boolean types, found {} and {}",
                        left, right
                    ))
                    .with_span(0, span)
                    .emit();
                    return Err(());
                }
                Ok(Type::Bool)
            }
            
            _ => Ok(Type::Unknown),
        }
    }
    
    fn infer_unary_op(&mut self, op: &ast::UnaryOperator, operand: &Type, span: Span) -> Result<Type, ()> {
        use ast::UnaryOperator::*;
        
        match op {
            Not => {
                if !operand.is_compatible_with(&Type::Bool) {
                    self.diagnostics.error(format!(
                        "Logical not requires boolean type, found {}",
                        operand
                    ))
                    .with_span(0, span)
                    .emit();
                    return Err(());
                }
                Ok(Type::Bool)
            }
            
            Negate => {
                if !operand.is_numeric() {
                    self.diagnostics.error(format!(
                        "Negation requires numeric type, found {}",
                        operand
                    ))
                    .with_span(0, span)
                    .emit();
                    return Err(());
                }
                Ok(operand.clone())
            }
        }
    }
    
    fn infer_member_access(&mut self, obj_type: &Type, member: &str, span: Span) -> Result<Type, ()> {
        // Handle built-in members
        match (obj_type, member) {
            (Type::Array(_), "length") => Ok(Type::Int),
            (Type::String, "length") => Ok(Type::Int),
            (Type::Uncertain(inner), "value") => Ok((**inner).clone()),
            (Type::Uncertain(_), "confidence") => Ok(Type::Float),
            
            _ => {
                // Check struct fields
                if let Type::Struct(s) = obj_type {
                    s.fields.get(member).cloned().ok_or_else(|| {
                        self.diagnostics.error(format!(
                            "Type {} has no field '{}'",
                            obj_type, member
                        ))
                        .with_span(0, span)
                        .emit();
                    })
                } else {
                    self.diagnostics.error(format!(
                        "Type {} has no member '{}'",
                        obj_type, member
                    ))
                    .with_span(0, span)
                    .emit();
                    Err(())
                }
            }
        }
    }
    
    fn verify_constraints(&mut self, program: &Program) -> Result<(), ()> {
        // Verify no unreachable code
        // Verify all paths return in non-void functions
        // Verify no use of uninitialized variables
        // etc.
        Ok(())
    }
    
    // Helper functions to convert AST types to semantic types
    
    fn ast_type_to_type(&self, ast_type: &ast::Type) -> Type {
        match ast_type {
            ast::Type::Simple(name) => {
                match name.as_str() {
                    "bool" => Type::Bool,
                    "int" => Type::Int,
                    "float" => Type::Float,
                    "string" => Type::String,
                    "void" => Type::Unit,
                    "any" => Type::Any,
                    name => {
                        // Check if it's a user-defined type
                        if self.env.lookup_type(name).is_some() {
                            Type::Struct(crate::types::StructType {
                                name: name.to_string(),
                                fields: HashMap::new(),
                                type_params: Vec::new(),
                                implements: Vec::new(),
                            })
                        } else {
                            Type::Unknown
                        }
                    }
                }
            }
            
            ast::Type::Array(elem) => {
                Type::Array(Box::new(self.ast_type_to_type(elem)))
            }
            
            ast::Type::Optional(inner) => {
                Type::Optional(Box::new(self.ast_type_to_type(inner)))
            }
            
            ast::Type::Uncertain(inner) => {
                Type::Uncertain(Box::new(self.ast_type_to_type(inner)))
            }
            
            ast::Type::Generic { name, args } => {
                let type_args = args.iter()
                    .map(|a| self.ast_type_to_type(a))
                    .collect();
                Type::Generic(name.clone(), type_args)
            }
            
            _ => Type::Unknown,
        }
    }
    
    fn function_signature_to_type(&self, func: &ast::Function) -> FunctionType {
        let params = func.parameters.iter()
            .map(|p| self.ast_type_to_type(&p.type_annotation))
            .collect();
        
        let return_type = func.return_type.as_ref()
            .map(|t| Box::new(self.ast_type_to_type(t)))
            .unwrap_or_else(|| Box::new(Type::Unit));
        
        let mut effects = Vec::new();
        
        // Analyze function body for effects
        if func.modifiers.contains(&ast::Modifier::Async) {
            effects.push(Effect::IO);
        }
        
        // Check for AI operations in function
        if self.function_uses_ai(func) {
            effects.push(Effect::AI);
        }
        
        FunctionType {
            params,
            return_type,
            is_async: func.modifiers.contains(&ast::Modifier::Async),
            is_pure: effects.is_empty(),
            effects,
        }
    }
    
    fn function_uses_ai(&self, _func: &ast::Function) -> bool {
        // TODO: Analyze function body for AI operations
        false
    }
    
    fn struct_to_type(&self, struct_decl: &ast::Struct) -> crate::types::StructType {
        let fields = struct_decl.fields.iter()
            .map(|f| (f.name.value.clone(), self.ast_type_to_type(&f.type_annotation)))
            .collect();
        
        crate::types::StructType {
            name: struct_decl.name.value.clone(),
            fields,
            type_params: struct_decl.type_params.clone(),
            implements: struct_decl.implements.clone(),
        }
    }
    
    fn enum_to_type(&self, enum_decl: &ast::Enum) -> crate::types::EnumType {
        let variants = enum_decl.variants.iter()
            .map(|v| {
                let variant_type = v.data.as_ref()
                    .map(|t| self.ast_type_to_type(t));
                (v.name.value.clone(), variant_type)
            })
            .collect();
        
        crate::types::EnumType {
            name: enum_decl.name.value.clone(),
            variants,
            type_params: enum_decl.type_params.clone(),
        }
    }
    
    fn interface_to_type(&self, interface_decl: &ast::Interface) -> crate::types::InterfaceType {
        let methods = interface_decl.methods.iter()
            .map(|m| {
                let func_type = self.function_signature_to_type(m);
                (m.name.value.clone(), func_type)
            })
            .collect();
        
        crate::types::InterfaceType {
            name: interface_decl.name.value.clone(),
            methods,
            type_params: interface_decl.type_params.clone(),
            extends: interface_decl.extends.clone(),
        }
    }
}