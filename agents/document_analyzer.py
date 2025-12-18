#!/usr/bin/env python3
"""
Document Analyzer Agent
Specialized agent for document analysis, summarization, and processing
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_agent import BaseAgent

class DocumentAnalyzerAgent(BaseAgent):
    """Agent specialized in document analysis and extraction"""
    
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        
        # Document processing configuration
        self.supported_extensions = ['.txt', '.rtf', '.md', '.docx']
        self.max_document_size = 10 * 1024 * 1024  # 10MB
        self.chunk_size = 2000  # Characters per chunk for large documents
        
        # Analysis types
        self.analysis_types = {
            'summary': self._generate_summary,
            'extract_entities': self._extract_entities,
            'classify': self._classify_document,
            'extract_keywords': self._extract_keywords,
            'sentiment': self._analyze_sentiment,
            'structure': self._analyze_structure,
            'compare': self._compare_documents,
            'search': self._search_documents
        }
        
        self.logger.info("DocumentAnalyzerAgent initialized")

    async def process_task(self, task) -> Dict[str, Any]:
        """Process document analysis task"""
        self.logger.info(f"Processing document analysis task: {task.description}")
        
        try:
            input_data = task.input_data
            analysis_type = input_data.get('analysis_type', 'summary')
            
            # Validate analysis type
            if analysis_type not in self.analysis_types:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # Get analysis function
            analysis_func = self.analysis_types[analysis_type]
            
            # Perform analysis
            result = await analysis_func(input_data)
            
            return self.format_response(
                content=result,
                metadata={
                    'analysis_type': analysis_type,
                    'processed_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return self.handle_error(e, "Document analysis failed")

    async def _generate_summary(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document summary"""
        document_path = input_data.get('document_path')
        documents = input_data.get('documents', [])
        summary_length = input_data.get('summary_length', 'medium')
        
        if document_path:
            # Single document summary
            content = self._read_document_safe(document_path)
            summary = await self._summarize_content(content, summary_length)
            
            return {
                'type': 'single_document_summary',
                'document_path': document_path,
                'summary': summary,
                'word_count': len(content.split()),
                'summary_ratio': len(summary.split()) / len(content.split()) if content else 0
            }
            
        elif documents:
            # Multiple document summary
            summaries = []
            for doc_path in documents:
                content = self._read_document_safe(doc_path)
                if content:
                    summary = await self._summarize_content(content, 'short')
                    summaries.append({
                        'document': doc_path,
                        'summary': summary,
                        'word_count': len(content.split())
                    })
            
            # Generate combined summary
            combined_content = "\n\n".join([s['summary'] for s in summaries])
            overall_summary = await self._summarize_content(combined_content, summary_length)
            
            return {
                'type': 'multi_document_summary',
                'individual_summaries': summaries,
                'overall_summary': overall_summary,
                'total_documents': len(summaries)
            }
        else:
            raise ValueError("Either document_path or documents list must be provided")

    async def _summarize_content(self, content: str, length: str = 'medium') -> str:
        """Summarize content using AI model"""
        if not content:
            return "No content to summarize"
        
        # Determine summary parameters based on length
        length_params = {
            'short': {'sentences': 2, 'max_tokens': 100},
            'medium': {'sentences': 5, 'max_tokens': 250},
            'long': {'sentences': 10, 'max_tokens': 500}
        }
        
        params = length_params.get(length, length_params['medium'])
        
        # Chunk content if too long
        if len(content) > self.chunk_size:
            chunks = self._chunk_content(content)
            chunk_summaries = []
            
            for chunk in chunks:
                prompt = self._build_summary_prompt(chunk, params['sentences'])
                summary = await self.call_lm_studio(prompt, max_tokens=params['max_tokens'])
                if summary and not summary.startswith('Error:'):
                    chunk_summaries.append(summary)
            
            # Combine chunk summaries
            if chunk_summaries:
                combined_content = "\n".join(chunk_summaries)
                final_prompt = self._build_summary_prompt(combined_content, params['sentences'])
                return await self.call_lm_studio(final_prompt, max_tokens=params['max_tokens'])
            else:
                return "Unable to generate summary"
        else:
            prompt = self._build_summary_prompt(content, params['sentences'])
            return await self.call_lm_studio(prompt, max_tokens=params['max_tokens'])

    def _build_summary_prompt(self, content: str, sentences: int) -> str:
        """Build prompt for summarization"""
        return f"""
Please provide a concise summary of the following text in approximately {sentences} sentences. 
Focus on the main points, key insights, and important conclusions.

Text to summarize:
{content[:1500]}...

Summary:"""

    async def _extract_entities(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from document(s)"""
        document_path = input_data.get('document_path')
        entity_types = input_data.get('entity_types', ['people', 'organizations', 'locations', 'dates', 'amounts'])
        
        if not document_path:
            raise ValueError("document_path required for entity extraction")
        
        content = self._read_document_safe(document_path)
        
        # Use AI model for entity extraction
        prompt = f"""
Extract the following types of entities from this text: {', '.join(entity_types)}

Text:
{content[:2000]}...

Please format the response as JSON with entity types as keys and lists of found entities as values.
Example: {{"people": ["John Smith", "Jane Doe"], "organizations": ["ABC Corp"], "locations": ["New York"]}}

Entities:"""

        response = await self.call_lm_studio(prompt, max_tokens=300)
        
        try:
            # Try to parse JSON response
            entities = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to simple regex extraction
            entities = self._extract_entities_regex(content)
        
        return {
            'document_path': document_path,
            'entities': entities,
            'extraction_method': 'ai_model' if isinstance(entities, dict) else 'regex_fallback'
        }

    def _extract_entities_regex(self, content: str) -> Dict[str, List[str]]:
        """Fallback entity extraction using regex patterns"""
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'amounts': []
        }
        
        # Simple regex patterns (basic implementation)
        patterns = {
            'dates': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            'amounts': r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities

    async def _classify_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify document type and content"""
        document_path = input_data.get('document_path')
        classification_scheme = input_data.get('classification_scheme', 'general')
        
        if not document_path:
            raise ValueError("document_path required for document classification")
        
        content = self._read_document_safe(document_path)
        
        # Define classification schemes
        schemes = {
            'general': ['technical', 'business', 'personal', 'legal', 'academic', 'creative'],
            'business': ['proposal', 'report', 'contract', 'memo', 'presentation', 'financial'],
            'technical': ['documentation', 'specification', 'manual', 'code', 'research', 'analysis']
        }
        
        categories = schemes.get(classification_scheme, schemes['general'])
        
        prompt = f"""
Classify this document into one of these categories: {', '.join(categories)}

Document content:
{content[:1500]}...

Provide your classification and a brief explanation (1-2 sentences) of why you chose this category.

Classification:"""

        response = await self.call_lm_studio(prompt, max_tokens=150)
        
        # Extract classification and confidence
        classification_result = self._parse_classification_response(response, categories)
        
        return {
            'document_path': document_path,
            'classification_scheme': classification_scheme,
            'classification': classification_result['category'],
            'confidence': classification_result['confidence'],
            'explanation': classification_result['explanation'],
            'available_categories': categories
        }

    def _parse_classification_response(self, response: str, categories: List[str]) -> Dict[str, Any]:
        """Parse classification response from AI model"""
        response_lower = response.lower()
        
        # Find mentioned category
        found_category = None
        for category in categories:
            if category.lower() in response_lower:
                found_category = category
                break
        
        # Estimate confidence based on response content
        confidence_indicators = ['confident', 'clearly', 'definitely', 'obviously']
        uncertainty_indicators = ['might', 'possibly', 'perhaps', 'could be']
        
        confidence = 0.7  # Default
        if any(indicator in response_lower for indicator in confidence_indicators):
            confidence = 0.9
        elif any(indicator in response_lower for indicator in uncertainty_indicators):
            confidence = 0.5
        
        return {
            'category': found_category or 'unknown',
            'confidence': confidence,
            'explanation': response.strip()
        }

    async def _extract_keywords(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract keywords and key phrases from document"""
        document_path = input_data.get('document_path')
        max_keywords = input_data.get('max_keywords', 20)
        
        if not document_path:
            raise ValueError("document_path required for keyword extraction")
        
        content = self._read_document_safe(document_path)
        
        prompt = f"""
Extract the {max_keywords} most important keywords and key phrases from this document. 
Focus on technical terms, important concepts, and significant topics.

Document content:
{content[:2000]}...

Please provide the keywords as a comma-separated list, ordered by importance.

Keywords:"""

        response = await self.call_lm_studio(prompt, max_tokens=200)
        
        # Parse keywords from response
        keywords = self._parse_keywords_response(response)
        
        # Also extract using simple frequency analysis as backup
        frequency_keywords = self._extract_keywords_frequency(content, max_keywords)
        
        return {
            'document_path': document_path,
            'ai_keywords': keywords,
            'frequency_keywords': frequency_keywords,
            'combined_keywords': list(set(keywords + frequency_keywords))[:max_keywords]
        }

    def _parse_keywords_response(self, response: str) -> List[str]:
        """Parse keywords from AI response"""
        # Clean and split the response
        keywords = []
        lines = response.strip().split('\n')
        
        for line in lines:
            # Remove common prefixes
            line = re.sub(r'^(keywords?:?\s*)', '', line, flags=re.IGNORECASE)
            
            # Split by commas and clean
            if ',' in line:
                keywords.extend([kw.strip() for kw in line.split(',') if kw.strip()])
            elif line.strip():
                keywords.append(line.strip())
        
        return keywords[:20]  # Limit to 20 keywords

    def _extract_keywords_frequency(self, content: str, max_keywords: int) -> List[str]:
        """Extract keywords using frequency analysis"""
        # Simple frequency-based keyword extraction
        import collections
        
        # Clean content
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'must', 'shall', 'not', 'no', 'yes', 'all', 'any',
            'some', 'many', 'much', 'more', 'most', 'other', 'another', 'such', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'than', 'then', 'now', 'here',
            'there', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'once', 'during', 'before', 'after', 'above', 'below', 'between', 'through',
            'into', 'from', 'about', 'against', 'very', 'too', 'only', 'own', 'same',
            'so', 'just', 'now', 'each', 'few', 'both', 'either', 'neither', 'also'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequencies
        word_freq = collections.Counter(filtered_words)
        
        # Return most common words
        return [word for word, count in word_freq.most_common(max_keywords)]

    async def _analyze_sentiment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document sentiment"""
        document_path = input_data.get('document_path')
        
        if not document_path:
            raise ValueError("document_path required for sentiment analysis")
        
        content = self._read_document_safe(document_path)
        
        prompt = f"""
Analyze the sentiment and tone of this document. Consider:
1. Overall emotional tone (positive, negative, neutral)
2. Formality level (formal, informal, mixed)
3. Confidence level (confident, uncertain, mixed)
4. Urgency level (urgent, normal, relaxed)

Document content:
{content[:2000]}...

Please provide your analysis in this format:
Sentiment: [positive/negative/neutral]
Formality: [formal/informal/mixed]
Confidence: [confident/uncertain/mixed]
Urgency: [urgent/normal/relaxed]
Explanation: [brief explanation]

Analysis:"""

        response = await self.call_lm_studio(prompt, max_tokens=200)
        
        # Parse sentiment analysis
        sentiment_result = self._parse_sentiment_response(response)
        
        return {
            'document_path': document_path,
            'sentiment_analysis': sentiment_result
        }

    def _parse_sentiment_response(self, response: str) -> Dict[str, str]:
        """Parse sentiment analysis response"""
        result = {
            'sentiment': 'neutral',
            'formality': 'mixed',
            'confidence': 'mixed',
            'urgency': 'normal',
            'explanation': response
        }
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip().lower()
            if line.startswith('sentiment:'):
                result['sentiment'] = line.split(':', 1)[1].strip()
            elif line.startswith('formality:'):
                result['formality'] = line.split(':', 1)[1].strip()
            elif line.startswith('confidence:'):
                result['confidence'] = line.split(':', 1)[1].strip()
            elif line.startswith('urgency:'):
                result['urgency'] = line.split(':', 1)[1].strip()
            elif line.startswith('explanation:'):
                result['explanation'] = line.split(':', 1)[1].strip()
        
        return result

    async def _analyze_structure(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document structure"""
        document_path = input_data.get('document_path')
        
        if not document_path:
            raise ValueError("document_path required for structure analysis")
        
        content = self._read_document_safe(document_path)
        
        # Analyze structure
        structure = {
            'total_characters': len(content),
            'total_words': len(content.split()),
            'total_lines': len(content.split('\n')),
            'paragraphs': len([p for p in content.split('\n\n') if p.strip()]),
            'sentences': len(re.findall(r'[.!?]+', content)),
            'avg_words_per_sentence': 0,
            'avg_sentences_per_paragraph': 0,
            'headings': [],
            'bullet_points': 0,
            'numbered_lists': 0
        }
        
        # Calculate averages
        if structure['sentences'] > 0:
            structure['avg_words_per_sentence'] = structure['total_words'] / structure['sentences']
        if structure['paragraphs'] > 0:
            structure['avg_sentences_per_paragraph'] = structure['sentences'] / structure['paragraphs']
        
        # Extract headings (simple patterns)
        heading_patterns = [
            r'^#+\s+(.+)$',  # Markdown headings
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS lines
            r'^\d+\.\s+([A-Z].+)$'  # Numbered headings
        ]
        
        for pattern in heading_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            structure['headings'].extend(matches)
        
        # Count lists
        structure['bullet_points'] = len(re.findall(r'^\s*[-*•]\s+', content, re.MULTILINE))
        structure['numbered_lists'] = len(re.findall(r'^\s*\d+\.\s+', content, re.MULTILINE))
        
        return {
            'document_path': document_path,
            'structure_analysis': structure
        }

    async def _compare_documents(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple documents"""
        documents = input_data.get('documents', [])
        comparison_type = input_data.get('comparison_type', 'similarity')
        
        if len(documents) < 2:
            raise ValueError("At least 2 documents required for comparison")
        
        # Read all documents
        doc_contents = {}
        for doc_path in documents:
            doc_contents[doc_path] = self._read_document_safe(doc_path)
        
        if comparison_type == 'similarity':
            return await self._compare_similarity(doc_contents)
        elif comparison_type == 'differences':
            return await self._compare_differences(doc_contents)
        else:
            raise ValueError(f"Unsupported comparison type: {comparison_type}")

    async def _compare_similarity(self, doc_contents: Dict[str, str]) -> Dict[str, Any]:
        """Compare document similarity"""
        documents = list(doc_contents.keys())
        
        # Use AI to analyze similarity
        combined_content = ""
        for i, (doc_path, content) in enumerate(doc_contents.items()):
            combined_content += f"\nDocument {i+1} ({Path(doc_path).name}):\n{content[:1000]}...\n"
        
        prompt = f"""
Compare these documents for similarity. Analyze:
1. Common themes and topics
2. Similar language and terminology
3. Overlapping content
4. Overall similarity score (0-100%)

{combined_content}

Provide a similarity analysis:"""

        response = await self.call_lm_studio(prompt, max_tokens=300)
        
        return {
            'comparison_type': 'similarity',
            'documents': documents,
            'similarity_analysis': response
        }

    async def _compare_differences(self, doc_contents: Dict[str, str]) -> Dict[str, Any]:
        """Compare document differences"""
        documents = list(doc_contents.keys())
        
        # Use AI to analyze differences
        combined_content = ""
        for i, (doc_path, content) in enumerate(doc_contents.items()):
            combined_content += f"\nDocument {i+1} ({Path(doc_path).name}):\n{content[:1000]}...\n"
        
        prompt = f"""
Compare these documents for differences. Analyze:
1. Unique topics in each document
2. Different perspectives or approaches
3. Contrasting information
4. Key differentiators

{combined_content}

Provide a differences analysis:"""

        response = await self.call_lm_studio(prompt, max_tokens=300)
        
        return {
            'comparison_type': 'differences',
            'documents': documents,
            'differences_analysis': response
        }

    async def _search_documents(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search through documents"""
        query = input_data.get('query', '')
        search_path = input_data.get('search_path', str(self.text_vault_path))
        max_results = input_data.get('max_results', 10)
        
        if not query:
            raise ValueError("Query required for document search")
        
        # Get list of documents to search
        documents = self.list_documents(search_path)
        
        # Search through documents
        results = []
        for doc_path in documents[:50]:  # Limit to 50 documents for performance
            try:
                content = self._read_document_safe(doc_path)
                if query.lower() in content.lower():
                    # Extract context around matches
                    matches = self._extract_search_context(content, query)
                    if matches:
                        results.append({
                            'document': doc_path,
                            'matches': matches,
                            'match_count': len(matches)
                        })
            except Exception as e:
                self.logger.warning(f"Error searching document {doc_path}: {e}")
        
        # Sort by relevance (number of matches)
        results.sort(key=lambda x: x['match_count'], reverse=True)
        
        return {
            'query': query,
            'search_path': search_path,
            'total_documents_searched': len(documents),
            'results': results[:max_results],
            'total_matches': len(results)
        }

    def _extract_search_context(self, content: str, query: str, context_chars: int = 200) -> List[Dict[str, str]]:
        """Extract context around search matches"""
        matches = []
        query_lower = query.lower()
        content_lower = content.lower()
        
        start = 0
        while True:
            pos = content_lower.find(query_lower, start)
            if pos == -1:
                break
            
            # Extract context
            context_start = max(0, pos - context_chars // 2)
            context_end = min(len(content), pos + len(query) + context_chars // 2)
            context = content[context_start:context_end]
            
            matches.append({
                'position': pos,
                'context': context.strip(),
                'match_text': content[pos:pos + len(query)]
            })
            
            start = pos + 1
        
        return matches

    def _read_document_safe(self, document_path: str) -> str:
        """Safely read document with error handling"""
        try:
            path = Path(document_path)
            
            # Check file size
            if path.stat().st_size > self.max_document_size:
                self.logger.warning(f"Document {document_path} exceeds size limit")
                return ""
            
            # Handle different file types
            if path.suffix.lower() == '.rtf':
                return self.extract_text_from_rtf(document_path)
            else:
                return self.read_document(document_path)
                
        except Exception as e:
            self.logger.error(f"Error reading document {document_path}: {e}")
            return ""

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into manageable chunks"""
        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks