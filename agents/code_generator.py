#!/usr/bin/env python3
"""
Code Generator Agent
Specialized agent for code generation and programming tasks
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_agent import BaseAgent

class CodeGeneratorAgent(BaseAgent):
    """Agent specialized in code generation and programming tasks"""
    
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        
        # Code generation configuration
        self.supported_languages = {
            'python': {'extension': '.py', 'model': 'Phind-CodeLlama-34B-v2'},
            'javascript': {'extension': '.js', 'model': 'Phind-CodeLlama-34B-v2'},
            'typescript': {'extension': '.ts', 'model': 'Phind-CodeLlama-34B-v2'},
            'java': {'extension': '.java', 'model': 'WizardCoder-15B-V1.0'},
            'cpp': {'extension': '.cpp', 'model': 'WizardCoder-15B-V1.0'},
            'csharp': {'extension': '.cs', 'model': 'WizardCoder-15B-V1.0'},
            'go': {'extension': '.go', 'model': 'WizardCoder-15B-V1.0'},
            'rust': {'extension': '.rs', 'model': 'WizardCoder-15B-V1.0'},
            'sql': {'extension': '.sql', 'model': 'Phind-CodeLlama-34B-v2'},
            'html': {'extension': '.html', 'model': 'Phind-CodeLlama-34B-v2'},
            'css': {'extension': '.css', 'model': 'Phind-CodeLlama-34B-v2'}
        }
        
        # Code generation types
        self.generation_types = {
            'function': self._generate_function,
            'class': self._generate_class,
            'script': self._generate_script,
            'api': self._generate_api,
            'test': self._generate_test,
            'documentation': self._generate_documentation,
            'refactor': self._refactor_code,
            'review': self._review_code,
            'debug': self._debug_code,
            'optimize': self._optimize_code
        }
        
        self.logger.info("CodeGeneratorAgent initialized")

    async def process_task(self, task) -> Dict[str, Any]:
        """Process code generation task"""
        self.logger.info(f"Processing code generation task: {task.description}")
        
        try:
            input_data = task.input_data
            generation_type = input_data.get('generation_type', 'function')
            
            # Validate generation type
            if generation_type not in self.generation_types:
                raise ValueError(f"Unsupported generation type: {generation_type}")
            
            # Get generation function
            generation_func = self.generation_types[generation_type]
            
            # Perform code generation
            result = await generation_func(input_data)
            
            return self.format_response(
                content=result,
                metadata={
                    'generation_type': generation_type,
                    'generated_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return self.handle_error(e, "Code generation failed")

    async def _generate_function(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a function based on requirements"""
        language = input_data.get('language', 'python')
        function_name = input_data.get('function_name', 'generated_function')
        requirements = input_data.get('requirements', '')
        parameters = input_data.get('parameters', [])
        return_type = input_data.get('return_type', '')
        
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        # Build function generation prompt
        prompt = self._build_function_prompt(
            language, function_name, requirements, parameters, return_type
        )
        
        # Generate code using appropriate model
        model = self.supported_languages[language]['model']
        code = await self.call_lm_studio(prompt, model=model, max_tokens=800)
        
        # Clean and format the code
        formatted_code = self._format_code(code, language)
        
        return {
            'type': 'function',
            'language': language,
            'function_name': function_name,
            'code': formatted_code,
            'requirements': requirements,
            'parameters': parameters,
            'return_type': return_type
        }

    def _build_function_prompt(self, language: str, function_name: str, 
                              requirements: str, parameters: List[str], 
                              return_type: str) -> str:
        """Build prompt for function generation"""
        
        param_str = ', '.join(parameters) if parameters else ''
        return_str = f" -> {return_type}" if return_type else ""
        
        prompt = f"""
Generate a {language} function with the following specifications:

Function Name: {function_name}
Parameters: {param_str}
Return Type: {return_type}

Requirements:
{requirements}

Please provide:
1. Clean, well-commented code
2. Proper error handling
3. Type hints (if applicable for the language)
4. Docstring/documentation

Generate only the function code, no additional text:
"""
        
        return prompt

    async def _generate_class(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a class based on requirements"""
        language = input_data.get('language', 'python')
        class_name = input_data.get('class_name', 'GeneratedClass')
        requirements = input_data.get('requirements', '')
        methods = input_data.get('methods', [])
        attributes = input_data.get('attributes', [])
        inheritance = input_data.get('inheritance', [])
        
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        # Build class generation prompt
        prompt = self._build_class_prompt(
            language, class_name, requirements, methods, attributes, inheritance
        )
        
        # Generate code using appropriate model
        model = self.supported_languages[language]['model']
        code = await self.call_lm_studio(prompt, model=model, max_tokens=1200)
        
        # Clean and format the code
        formatted_code = self._format_code(code, language)
        
        return {
            'type': 'class',
            'language': language,
            'class_name': class_name,
            'code': formatted_code,
            'requirements': requirements,
            'methods': methods,
            'attributes': attributes,
            'inheritance': inheritance
        }

    def _build_class_prompt(self, language: str, class_name: str, 
                           requirements: str, methods: List[str], 
                           attributes: List[str], inheritance: List[str]) -> str:
        """Build prompt for class generation"""
        
        inheritance_str = f"({', '.join(inheritance)})" if inheritance else ""
        methods_str = '\n'.join([f"- {method}" for method in methods]) if methods else "- Basic methods as needed"
        attributes_str = '\n'.join([f"- {attr}" for attr in attributes]) if attributes else "- Attributes as needed"
        
        prompt = f"""
Generate a {language} class with the following specifications:

Class Name: {class_name}{inheritance_str}

Requirements:
{requirements}

Required Methods:
{methods_str}

Required Attributes:
{attributes_str}

Please provide:
1. Clean, well-structured code
2. Proper constructor/initialization
3. Appropriate access modifiers
4. Documentation for the class and methods
5. Error handling where appropriate

Generate only the class code, no additional text:
"""
        
        return prompt

    async def _generate_script(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete script"""
        language = input_data.get('language', 'python')
        script_purpose = input_data.get('purpose', 'utility script')
        requirements = input_data.get('requirements', '')
        dependencies = input_data.get('dependencies', [])
        
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        # Build script generation prompt
        prompt = self._build_script_prompt(
            language, script_purpose, requirements, dependencies
        )
        
        # Generate code using appropriate model
        model = self.supported_languages[language]['model']
        code = await self.call_lm_studio(prompt, model=model, max_tokens=1500)
        
        # Clean and format the code
        formatted_code = self._format_code(code, language)
        
        return {
            'type': 'script',
            'language': language,
            'purpose': script_purpose,
            'code': formatted_code,
            'requirements': requirements,
            'dependencies': dependencies
        }

    def _build_script_prompt(self, language: str, purpose: str, 
                            requirements: str, dependencies: List[str]) -> str:
        """Build prompt for script generation"""
        
        deps_str = '\n'.join([f"- {dep}" for dep in dependencies]) if dependencies else "- Standard library only"
        
        prompt = f"""
Generate a complete {language} script for the following purpose:

Purpose: {purpose}

Requirements:
{requirements}

Dependencies:
{deps_str}

Please provide:
1. Complete, executable script
2. Proper imports and setup
3. Main execution logic
4. Error handling
5. Comments and documentation
6. Command-line argument handling (if applicable)

Generate the complete script code:
"""
        
        return prompt

    async def _generate_api(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API code"""
        language = input_data.get('language', 'python')
        api_type = input_data.get('api_type', 'REST')
        endpoints = input_data.get('endpoints', [])
        framework = input_data.get('framework', 'FastAPI' if language == 'python' else 'Express')
        
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        # Build API generation prompt
        prompt = self._build_api_prompt(language, api_type, endpoints, framework)
        
        # Generate code using appropriate model
        model = self.supported_languages[language]['model']
        code = await self.call_lm_studio(prompt, model=model, max_tokens=1500)
        
        # Clean and format the code
        formatted_code = self._format_code(code, language)
        
        return {
            'type': 'api',
            'language': language,
            'api_type': api_type,
            'framework': framework,
            'code': formatted_code,
            'endpoints': endpoints
        }

    def _build_api_prompt(self, language: str, api_type: str, 
                         endpoints: List[str], framework: str) -> str:
        """Build prompt for API generation"""
        
        endpoints_str = '\n'.join([f"- {endpoint}" for endpoint in endpoints]) if endpoints else "- Basic CRUD endpoints"
        
        prompt = f"""
Generate a {api_type} API in {language} using {framework} with the following specifications:

Endpoints:
{endpoints_str}

Please provide:
1. Complete API server setup
2. Route definitions with proper HTTP methods
3. Request/response models
4. Error handling
5. Basic validation
6. Documentation/comments

Generate the API code:
"""
        
        return prompt

    async def _generate_test(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test code"""
        language = input_data.get('language', 'python')
        test_type = input_data.get('test_type', 'unit')
        code_to_test = input_data.get('code_to_test', '')
        test_framework = input_data.get('framework', 'pytest' if language == 'python' else 'jest')
        
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        # Build test generation prompt
        prompt = self._build_test_prompt(language, test_type, code_to_test, test_framework)
        
        # Generate code using appropriate model
        model = self.supported_languages[language]['model']
        code = await self.call_lm_studio(prompt, model=model, max_tokens=1000)
        
        # Clean and format the code
        formatted_code = self._format_code(code, language)
        
        return {
            'type': 'test',
            'language': language,
            'test_type': test_type,
            'framework': test_framework,
            'code': formatted_code,
            'code_to_test': code_to_test
        }

    def _build_test_prompt(self, language: str, test_type: str, 
                          code_to_test: str, framework: str) -> str:
        """Build prompt for test generation"""
        
        prompt = f"""
Generate {test_type} tests in {language} using {framework} for the following code:

Code to test:
{code_to_test}

Please provide:
1. Comprehensive test cases
2. Edge case testing
3. Error condition testing
4. Proper test setup and teardown
5. Clear test descriptions
6. Assertions for expected behavior

Generate the test code:
"""
        
        return prompt

    async def _generate_documentation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code documentation"""
        code = input_data.get('code', '')
        doc_type = input_data.get('doc_type', 'inline')
        language = input_data.get('language', 'python')
        
        # Build documentation prompt
        prompt = f"""
Generate {doc_type} documentation for the following {language} code:

Code:
{code}

Please provide:
1. Clear function/class descriptions
2. Parameter documentation
3. Return value documentation
4. Usage examples
5. Error conditions

Generate the documented code:
"""
        
        # Generate documentation
        model = self.supported_languages.get(language, {}).get('model', 'Phind-CodeLlama-34B-v2')
        documented_code = await self.call_lm_studio(prompt, model=model, max_tokens=1000)
        
        return {
            'type': 'documentation',
            'language': language,
            'doc_type': doc_type,
            'original_code': code,
            'documented_code': documented_code
        }

    async def _refactor_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refactor existing code"""
        code = input_data.get('code', '')
        refactor_goals = input_data.get('goals', ['improve readability', 'optimize performance'])
        language = input_data.get('language', 'python')
        
        # Build refactoring prompt
        goals_str = ', '.join(refactor_goals)
        prompt = f"""
Refactor the following {language} code with these goals: {goals_str}

Original code:
{code}

Please provide:
1. Improved code structure
2. Better variable/function names
3. Optimized algorithms where applicable
4. Removed code duplication
5. Comments explaining changes

Generate the refactored code:
"""
        
        # Generate refactored code
        model = self.supported_languages.get(language, {}).get('model', 'Phind-CodeLlama-34B-v2')
        refactored_code = await self.call_lm_studio(prompt, model=model, max_tokens=1200)
        
        return {
            'type': 'refactor',
            'language': language,
            'goals': refactor_goals,
            'original_code': code,
            'refactored_code': refactored_code
        }

    async def _review_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Review code for issues and improvements"""
        code = input_data.get('code', '')
        review_focus = input_data.get('focus', ['security', 'performance', 'maintainability'])
        language = input_data.get('language', 'python')
        
        # Build code review prompt
        focus_str = ', '.join(review_focus)
        prompt = f"""
Perform a code review for the following {language} code, focusing on: {focus_str}

Code to review:
{code}

Please provide:
1. Issues found (security, bugs, performance)
2. Improvement suggestions
3. Best practices recommendations
4. Code quality assessment
5. Specific line-by-line feedback

Generate the code review:
"""
        
        # Generate code review
        model = self.supported_languages.get(language, {}).get('model', 'Phind-CodeLlama-34B-v2')
        review = await self.call_lm_studio(prompt, model=model, max_tokens=800)
        
        return {
            'type': 'review',
            'language': language,
            'focus_areas': review_focus,
            'code': code,
            'review': review
        }

    async def _debug_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Debug code issues"""
        code = input_data.get('code', '')
        error_message = input_data.get('error_message', '')
        expected_behavior = input_data.get('expected_behavior', '')
        language = input_data.get('language', 'python')
        
        # Build debugging prompt
        prompt = f"""
Debug the following {language} code:

Code:
{code}

Error message:
{error_message}

Expected behavior:
{expected_behavior}

Please provide:
1. Root cause analysis
2. Fixed code
3. Explanation of the fix
4. Prevention strategies

Generate the debugging analysis and fixed code:
"""
        
        # Generate debugging solution
        model = self.supported_languages.get(language, {}).get('model', 'Phind-CodeLlama-34B-v2')
        debug_solution = await self.call_lm_studio(prompt, model=model, max_tokens=1000)
        
        return {
            'type': 'debug',
            'language': language,
            'original_code': code,
            'error_message': error_message,
            'expected_behavior': expected_behavior,
            'solution': debug_solution
        }

    async def _optimize_code(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize code for performance"""
        code = input_data.get('code', '')
        optimization_goals = input_data.get('goals', ['speed', 'memory'])
        language = input_data.get('language', 'python')
        
        # Build optimization prompt
        goals_str = ', '.join(optimization_goals)
        prompt = f"""
Optimize the following {language} code for: {goals_str}

Code to optimize:
{code}

Please provide:
1. Optimized code
2. Performance improvements made
3. Trade-offs considered
4. Benchmarking suggestions
5. Alternative approaches

Generate the optimized code:
"""
        
        # Generate optimized code
        model = self.supported_languages.get(language, {}).get('model', 'Phind-CodeLlama-34B-v2')
        optimized_code = await self.call_lm_studio(prompt, model=model, max_tokens=1200)
        
        return {
            'type': 'optimize',
            'language': language,
            'optimization_goals': optimization_goals,
            'original_code': code,
            'optimized_code': optimized_code
        }

    def _format_code(self, code: str, language: str) -> str:
        """Clean and format generated code"""
        if not code:
            return ""
        
        # Remove common AI response prefixes/suffixes
        code = re.sub(r'^(here\'s|here is|this is).*?code:?\s*', '', code, flags=re.IGNORECASE)
        code = re.sub(r'```\w*\n?', '', code)  # Remove code block markers
        code = re.sub(r'\n\s*```\s*$', '', code)  # Remove trailing code block markers
        
        # Clean up extra whitespace
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)