#!/usr/bin/env python3
"""
Data Processor Agent
Specialized agent for data processing and analysis tasks
"""

import asyncio
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_agent import BaseAgent

class DataProcessorAgent(BaseAgent):
    """Agent for data processing and analysis tasks"""
    
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        
        # Data processing types
        self.processing_types = {
            'analyze': self._analyze_data,
            'clean': self._clean_data,
            'transform': self._transform_data,
            'aggregate': self._aggregate_data,
            'visualize': self._visualize_data,
            'extract': self._extract_data,
            'validate': self._validate_data,
            'merge': self._merge_data,
            'filter': self._filter_data,
            'summarize': self._summarize_data
        }
        
        # Supported data formats
        self.supported_formats = {
            '.csv': self._process_csv,
            '.json': self._process_json,
            '.txt': self._process_text,
            '.log': self._process_log
        }
        
        self.logger.info("DataProcessorAgent initialized")

    async def process_task(self, task) -> Dict[str, Any]:
        """Process data processing task"""
        self.logger.info(f"Processing data task: {task.description}")
        
        try:
            input_data = task.input_data
            processing_type = input_data.get('processing_type', 'analyze')
            
            # Validate processing type
            if processing_type not in self.processing_types:
                raise ValueError(f"Unsupported processing type: {processing_type}")
            
            # Get processing function
            processing_func = self.processing_types[processing_type]
            
            # Process data
            result = await processing_func(input_data)
            
            return self.format_response(
                content=result,
                metadata={
                    'processing_type': processing_type,
                    'processed_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return self.handle_error(e, "Data processing failed")

    async def _analyze_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data and provide insights"""
        data_path = input_data.get('data_path')
        analysis_type = input_data.get('analysis_type', 'general')
        
        if not data_path:
            raise ValueError("data_path required for data analysis")
        
        # Load and process data
        data = await self._load_data(data_path)
        
        # Perform analysis based on type
        if analysis_type == 'statistical':
            analysis = await self._statistical_analysis(data)
        elif analysis_type == 'pattern':
            analysis = await self._pattern_analysis(data)
        elif analysis_type == 'trend':
            analysis = await self._trend_analysis(data)
        else:
            analysis = await self._general_analysis(data)
        
        return {
            'data_path': data_path,
            'analysis_type': analysis_type,
            'analysis': analysis,
            'data_summary': self._get_data_summary(data)
        }

    async def _clean_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and preprocess data"""
        data_path = input_data.get('data_path')
        cleaning_operations = input_data.get('operations', ['remove_duplicates', 'handle_missing'])
        output_path = input_data.get('output_path')
        
        if not data_path:
            raise ValueError("data_path required for data cleaning")
        
        # Load data
        data = await self._load_data(data_path)
        
        # Apply cleaning operations
        cleaned_data = data
        operations_performed = []
        
        for operation in cleaning_operations:
            if operation == 'remove_duplicates':
                cleaned_data, removed_count = self._remove_duplicates(cleaned_data)
                operations_performed.append(f"Removed {removed_count} duplicates")
            elif operation == 'handle_missing':
                cleaned_data, filled_count = self._handle_missing_values(cleaned_data)
                operations_performed.append(f"Handled {filled_count} missing values")
            elif operation == 'normalize':
                cleaned_data = self._normalize_data(cleaned_data)
                operations_performed.append("Normalized data")
            elif operation == 'trim_whitespace':
                cleaned_data = self._trim_whitespace(cleaned_data)
                operations_performed.append("Trimmed whitespace")
        
        # Save cleaned data if output path provided
        if output_path:
            await self._save_data(cleaned_data, output_path)
        
        return {
            'original_data_path': data_path,
            'output_path': output_path,
            'operations_performed': operations_performed,
            'original_size': len(data) if isinstance(data, list) else 1,
            'cleaned_size': len(cleaned_data) if isinstance(cleaned_data, list) else 1,
            'cleaned_data_preview': self._get_data_preview(cleaned_data)
        }

    async def _transform_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data format or structure"""
        data_path = input_data.get('data_path')
        transformation_type = input_data.get('transformation_type', 'format_conversion')
        target_format = input_data.get('target_format', 'json')
        output_path = input_data.get('output_path')
        
        if not data_path:
            raise ValueError("data_path required for data transformation")
        
        # Load data
        data = await self._load_data(data_path)
        
        # Apply transformation
        if transformation_type == 'format_conversion':
            transformed_data = await self._convert_format(data, target_format)
        elif transformation_type == 'restructure':
            transformed_data = await self._restructure_data(data, input_data.get('structure_config', {}))
        elif transformation_type == 'aggregate':
            transformed_data = await self._aggregate_transform(data, input_data.get('group_by', []))
        else:
            raise ValueError(f"Unsupported transformation type: {transformation_type}")
        
        # Save transformed data if output path provided
        if output_path:
            await self._save_data(transformed_data, output_path)
        
        return {
            'original_data_path': data_path,
            'transformation_type': transformation_type,
            'target_format': target_format,
            'output_path': output_path,
            'transformed_data_preview': self._get_data_preview(transformed_data)
        }

    async def _aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data by specified criteria"""
        data_path = input_data.get('data_path')
        group_by = input_data.get('group_by', [])
        aggregation_functions = input_data.get('functions', ['count', 'sum', 'avg'])
        
        if not data_path:
            raise ValueError("data_path required for data aggregation")
        
        # Load data
        data = await self._load_data(data_path)
        
        # Perform aggregation
        aggregated_data = await self._perform_aggregation(data, group_by, aggregation_functions)
        
        return {
            'data_path': data_path,
            'group_by': group_by,
            'aggregation_functions': aggregation_functions,
            'aggregated_data': aggregated_data,
            'summary': self._get_aggregation_summary(aggregated_data)
        }

    async def _visualize_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data visualization descriptions"""
        data_path = input_data.get('data_path')
        chart_type = input_data.get('chart_type', 'auto')
        
        if not data_path:
            raise ValueError("data_path required for data visualization")
        
        # Load data
        data = await self._load_data(data_path)
        
        # Analyze data for visualization
        viz_recommendations = await self._recommend_visualizations(data, chart_type)
        
        return {
            'data_path': data_path,
            'requested_chart_type': chart_type,
            'visualization_recommendations': viz_recommendations,
            'data_characteristics': self._analyze_data_characteristics(data)
        }

    async def _extract_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specific data from source"""
        source_path = input_data.get('source_path')
        extraction_criteria = input_data.get('criteria', {})
        extraction_type = input_data.get('extraction_type', 'filter')
        
        if not source_path:
            raise ValueError("source_path required for data extraction")
        
        # Load source data
        source_data = await self._load_data(source_path)
        
        # Extract data based on criteria
        if extraction_type == 'filter':
            extracted_data = self._filter_by_criteria(source_data, extraction_criteria)
        elif extraction_type == 'select_columns':
            extracted_data = self._select_columns(source_data, extraction_criteria.get('columns', []))
        elif extraction_type == 'sample':
            extracted_data = self._sample_data(source_data, extraction_criteria.get('sample_size', 100))
        else:
            raise ValueError(f"Unsupported extraction type: {extraction_type}")
        
        return {
            'source_path': source_path,
            'extraction_type': extraction_type,
            'criteria': extraction_criteria,
            'original_size': len(source_data) if isinstance(source_data, list) else 1,
            'extracted_size': len(extracted_data) if isinstance(extracted_data, list) else 1,
            'extracted_data': self._get_data_preview(extracted_data)
        }

    async def _validate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and integrity"""
        data_path = input_data.get('data_path')
        validation_rules = input_data.get('rules', {})
        
        if not data_path:
            raise ValueError("data_path required for data validation")
        
        # Load data
        data = await self._load_data(data_path)
        
        # Perform validation
        validation_results = await self._perform_validation(data, validation_rules)
        
        return {
            'data_path': data_path,
            'validation_rules': validation_rules,
            'validation_results': validation_results,
            'is_valid': validation_results.get('overall_valid', False),
            'issues_found': validation_results.get('issues', [])
        }

    async def _merge_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple data sources"""
        data_sources = input_data.get('data_sources', [])
        merge_type = input_data.get('merge_type', 'inner')
        merge_keys = input_data.get('merge_keys', [])
        
        if len(data_sources) < 2:
            raise ValueError("At least 2 data sources required for merging")
        
        # Load all data sources
        loaded_data = []
        for source in data_sources:
            data = await self._load_data(source)
            loaded_data.append({'source': source, 'data': data})
        
        # Perform merge
        merged_data = await self._perform_merge(loaded_data, merge_type, merge_keys)
        
        return {
            'data_sources': data_sources,
            'merge_type': merge_type,
            'merge_keys': merge_keys,
            'original_sizes': [len(d['data']) if isinstance(d['data'], list) else 1 for d in loaded_data],
            'merged_size': len(merged_data) if isinstance(merged_data, list) else 1,
            'merged_data_preview': self._get_data_preview(merged_data)
        }

    async def _filter_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data based on conditions"""
        data_path = input_data.get('data_path')
        filter_conditions = input_data.get('conditions', {})
        
        if not data_path:
            raise ValueError("data_path required for data filtering")
        
        # Load data
        data = await self._load_data(data_path)
        
        # Apply filters
        filtered_data = self._apply_filters(data, filter_conditions)
        
        return {
            'data_path': data_path,
            'filter_conditions': filter_conditions,
            'original_size': len(data) if isinstance(data, list) else 1,
            'filtered_size': len(filtered_data) if isinstance(filtered_data, list) else 1,
            'filtered_data_preview': self._get_data_preview(filtered_data)
        }

    async def _summarize_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data summary and statistics"""
        data_path = input_data.get('data_path')
        summary_type = input_data.get('summary_type', 'comprehensive')
        
        if not data_path:
            raise ValueError("data_path required for data summarization")
        
        # Load data
        data = await self._load_data(data_path)
        
        # Generate summary
        if summary_type == 'basic':
            summary = self._basic_summary(data)
        elif summary_type == 'statistical':
            summary = await self._statistical_summary(data)
        else:
            summary = await self._comprehensive_summary(data)
        
        return {
            'data_path': data_path,
            'summary_type': summary_type,
            'summary': summary
        }

    # Helper methods for data operations

    async def _load_data(self, data_path: str) -> Any:
        """Load data from file"""
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Determine file type and load accordingly
        extension = path.suffix.lower()
        
        if extension in self.supported_formats:
            return await self.supported_formats[extension](data_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    async def _process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
        except Exception as e:
            self.logger.error(f"Error processing CSV {file_path}: {e}")
            raise
        return data

    async def _process_json(self, file_path: str) -> Any:
        """Process JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Error processing JSON {file_path}: {e}")
            raise
        return data

    async def _process_text(self, file_path: str) -> List[str]:
        """Process text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return [line.strip() for line in lines if line.strip()]
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {e}")
            raise

    async def _process_log(self, file_path: str) -> List[Dict[str, str]]:
        """Process log file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse log entries (basic implementation)
            log_entries = []
            for i, line in enumerate(lines):
                if line.strip():
                    log_entries.append({
                        'line_number': i + 1,
                        'content': line.strip(),
                        'timestamp': self._extract_timestamp(line),
                        'level': self._extract_log_level(line)
                    })
            
            return log_entries
        except Exception as e:
            self.logger.error(f"Error processing log file {file_path}: {e}")
            raise

    def _extract_timestamp(self, log_line: str) -> Optional[str]:
        """Extract timestamp from log line"""
        # Basic timestamp extraction patterns
        patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
            r'\w{3} \d{2} \d{2}:\d{2}:\d{2}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_line)
            if match:
                return match.group()
        
        return None

    def _extract_log_level(self, log_line: str) -> Optional[str]:
        """Extract log level from log line"""
        levels = ['ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE']
        line_upper = log_line.upper()
        
        for level in levels:
            if level in line_upper:
                return level
        
        return None

    async def _save_data(self, data: Any, output_path: str):
        """Save data to file"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        extension = path.suffix.lower()
        
        if extension == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        elif extension == '.csv' and isinstance(data, list) and data:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if isinstance(data[0], dict):
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                else:
                    writer = csv.writer(f)
                    writer.writerows(data)
        else:
            # Default to text format
            with open(output_path, 'w', encoding='utf-8') as f:
                if isinstance(data, (list, tuple)):
                    for item in data:
                        f.write(f"{item}\n")
                else:
                    f.write(str(data))

    def _get_data_summary(self, data: Any) -> Dict[str, Any]:
        """Get basic data summary"""
        if isinstance(data, list):
            return {
                'type': 'list',
                'size': len(data),
                'sample': data[:3] if data else []
            }
        elif isinstance(data, dict):
            return {
                'type': 'dict',
                'keys': list(data.keys())[:10],
                'size': len(data)
            }
        else:
            return {
                'type': type(data).__name__,
                'value': str(data)[:100]
            }

    def _get_data_preview(self, data: Any, max_items: int = 5) -> Any:
        """Get preview of data"""
        if isinstance(data, list):
            return data[:max_items]
        elif isinstance(data, dict):
            return dict(list(data.items())[:max_items])
        else:
            return data

    async def _statistical_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform statistical analysis"""
        if not isinstance(data, list):
            return {"error": "Statistical analysis requires list data"}
        
        # Basic statistical analysis
        analysis = {
            'count': len(data),
            'data_types': {},
            'summary': "Basic statistical analysis completed"
        }
        
        # Analyze data types if it's a list of dicts
        if data and isinstance(data[0], dict):
            for key in data[0].keys():
                values = [item.get(key) for item in data if key in item]
                numeric_values = []
                
                for val in values:
                    try:
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        pass
                
                if numeric_values:
                    analysis['data_types'][key] = {
                        'type': 'numeric',
                        'count': len(numeric_values),
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'avg': sum(numeric_values) / len(numeric_values)
                    }
                else:
                    analysis['data_types'][key] = {
                        'type': 'categorical',
                        'unique_values': len(set(str(v) for v in values)),
                        'sample_values': list(set(str(v) for v in values))[:5]
                    }
        
        return analysis

    async def _general_analysis(self, data: Any) -> str:
        """Perform general data analysis using AI"""
        data_summary = self._get_data_summary(data)
        
        prompt = f"""
Analyze this data and provide insights:

Data Summary: {json.dumps(data_summary, indent=2)}

Data Sample: {json.dumps(self._get_data_preview(data), indent=2)}

Please provide:
1. Data structure analysis
2. Key patterns or trends
3. Data quality assessment
4. Potential use cases
5. Recommendations for further analysis

Analysis:
"""
        
        analysis = await self.call_lm_studio(prompt, max_tokens=500)
        return analysis

    # Additional helper methods would be implemented here for:
    # - _remove_duplicates
    # - _handle_missing_values
    # - _normalize_data
    # - _trim_whitespace
    # - _convert_format
    # - _restructure_data
    # - _perform_aggregation
    # - _recommend_visualizations
    # - _analyze_data_characteristics
    # - _filter_by_criteria
    # - _select_columns
    # - _sample_data
    # - _perform_validation
    # - _perform_merge
    # - _apply_filters
    # - _basic_summary
    # - _statistical_summary
    # - _comprehensive_summary

    def _remove_duplicates(self, data: List[Any]) -> tuple[List[Any], int]:
        """Remove duplicate entries"""
        if not isinstance(data, list):
            return data, 0
        
        original_count = len(data)
        
        # For list of dicts, convert to JSON strings for comparison
        if data and isinstance(data[0], dict):
            seen = set()
            unique_data = []
            for item in data:
                item_str = json.dumps(item, sort_keys=True)
                if item_str not in seen:
                    seen.add(item_str)
                    unique_data.append(item)
        else:
            unique_data = list(set(data))
        
        removed_count = original_count - len(unique_data)
        return unique_data, removed_count

    def _handle_missing_values(self, data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
        """Handle missing values in data"""
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            return data, 0
        
        filled_count = 0
        processed_data = []
        
        for item in data:
            processed_item = item.copy()
            for key, value in processed_item.items():
                if value is None or value == '' or value == 'null':
                    # Simple strategy: replace with 'N/A' for strings, 0 for numbers
                    processed_item[key] = 'N/A'
                    filled_count += 1
            processed_data.append(processed_item)
        
        return processed_data, filled_count