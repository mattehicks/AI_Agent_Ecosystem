#!/usr/bin/env python3
"""
Research Assistant Agent
Specialized agent for research and information gathering
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_agent import BaseAgent

class ResearchAssistantAgent(BaseAgent):
    """Agent for research and information gathering"""
    
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        
        # Research configuration
        self.research_types = {
            'query': self._conduct_query_research,
            'comparative': self._conduct_comparative_research,
            'comprehensive': self._conduct_comprehensive_research,
            'fact_check': self._fact_check_research,
            'trend_analysis': self._trend_analysis_research,
            'literature_review': self._literature_review_research
        }
        
        # Source types
        self.source_types = {
            'documents': self._search_documents,
            'knowledge_base': self._search_knowledge_base,
            'privateGPT': self._query_privateGPT_research,
            'text_vault': self._search_text_vault,
            'prompts': self._search_prompts
        }
        
        self.logger.info("ResearchAssistantAgent initialized")

    async def process_task(self, task) -> Dict[str, Any]:
        """Process research task"""
        self.logger.info(f"Processing research task: {task.description}")
        
        try:
            input_data = task.input_data
            research_type = input_data.get('research_type', 'query')
            
            # Validate research type
            if research_type not in self.research_types:
                raise ValueError(f"Unsupported research type: {research_type}")
            
            # Get research function
            research_func = self.research_types[research_type]
            
            # Conduct research
            result = await research_func(input_data)
            
            return self.format_response(
                content=result,
                metadata={
                    'research_type': research_type,
                    'researched_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return self.handle_error(e, "Research failed")

    async def _conduct_query_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct basic query research"""
        query = input_data.get('query', '')
        sources = input_data.get('sources', ['documents', 'knowledge_base'])
        max_results = input_data.get('max_results', 10)
        
        if not query:
            raise ValueError("Query is required for research")
        
        research_results = []
        
        # Search each specified source
        for source in sources:
            if source in self.source_types:
                try:
                    source_results = await self.source_types[source](query, max_results)
                    research_results.append({
                        'source': source,
                        'results': source_results,
                        'result_count': len(source_results) if isinstance(source_results, list) else 1
                    })
                except Exception as e:
                    self.logger.warning(f"Error searching source {source}: {e}")
                    research_results.append({
                        'source': source,
                        'results': f"Error: {e}",
                        'result_count': 0
                    })
        
        # Synthesize findings
        synthesis = await self._synthesize_research_results(query, research_results)
        
        return {
            'query': query,
            'sources_searched': sources,
            'individual_results': research_results,
            'synthesis': synthesis,
            'total_results': sum(r['result_count'] for r in research_results)
        }

    async def _conduct_comparative_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comparative research between topics"""
        topics = input_data.get('topics', [])
        comparison_aspects = input_data.get('aspects', ['general'])
        sources = input_data.get('sources', ['documents', 'knowledge_base'])
        
        if len(topics) < 2:
            raise ValueError("At least 2 topics required for comparative research")
        
        # Research each topic
        topic_results = {}
        for topic in topics:
            topic_query = f"{topic} {' '.join(comparison_aspects)}"
            topic_research = await self._conduct_query_research({
                'query': topic_query,
                'sources': sources,
                'max_results': 5
            })
            topic_results[topic] = topic_research
        
        # Generate comparison
        comparison = await self._generate_comparison(topics, topic_results, comparison_aspects)
        
        return {
            'research_type': 'comparative',
            'topics': topics,
            'comparison_aspects': comparison_aspects,
            'individual_research': topic_results,
            'comparison': comparison
        }

    async def _conduct_comprehensive_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        topic = input_data.get('topic', '')
        research_depth = input_data.get('depth', 'medium')  # shallow, medium, deep
        sources = input_data.get('sources', ['documents', 'knowledge_base', 'privateGPT'])
        
        if not topic:
            raise ValueError("Topic is required for comprehensive research")
        
        # Define research queries based on depth
        depth_queries = {
            'shallow': [topic],
            'medium': [
                topic,
                f"{topic} overview",
                f"{topic} applications",
                f"{topic} benefits challenges"
            ],
            'deep': [
                topic,
                f"{topic} overview introduction",
                f"{topic} history background",
                f"{topic} applications use cases",
                f"{topic} benefits advantages",
                f"{topic} challenges limitations",
                f"{topic} future trends",
                f"{topic} best practices"
            ]
        }
        
        queries = depth_queries.get(research_depth, depth_queries['medium'])
        
        # Conduct research for each query
        comprehensive_results = []
        for query in queries:
            query_results = await self._conduct_query_research({
                'query': query,
                'sources': sources,
                'max_results': 8
            })
            comprehensive_results.append({
                'query': query,
                'results': query_results
            })
        
        # Generate comprehensive report
        report = await self._generate_comprehensive_report(topic, comprehensive_results, research_depth)
        
        return {
            'research_type': 'comprehensive',
            'topic': topic,
            'depth': research_depth,
            'queries_researched': queries,
            'detailed_results': comprehensive_results,
            'comprehensive_report': report
        }

    async def _fact_check_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fact-check claims or statements"""
        claims = input_data.get('claims', [])
        sources = input_data.get('sources', ['documents', 'knowledge_base'])
        
        if not claims:
            raise ValueError("Claims are required for fact-checking")
        
        fact_check_results = []
        
        for claim in claims:
            # Research the claim
            claim_research = await self._conduct_query_research({
                'query': claim,
                'sources': sources,
                'max_results': 5
            })
            
            # Analyze the claim
            analysis = await self._analyze_claim(claim, claim_research)
            
            fact_check_results.append({
                'claim': claim,
                'research': claim_research,
                'analysis': analysis
            })
        
        return {
            'research_type': 'fact_check',
            'claims': claims,
            'fact_check_results': fact_check_results
        }

    async def _trend_analysis_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in a topic or field"""
        topic = input_data.get('topic', '')
        time_period = input_data.get('time_period', 'recent')
        sources = input_data.get('sources', ['documents', 'knowledge_base'])
        
        if not topic:
            raise ValueError("Topic is required for trend analysis")
        
        # Define trend-focused queries
        trend_queries = [
            f"{topic} trends {time_period}",
            f"{topic} developments {time_period}",
            f"{topic} changes evolution",
            f"{topic} future predictions",
            f"{topic} market analysis"
        ]
        
        # Research trends
        trend_results = []
        for query in trend_queries:
            query_results = await self._conduct_query_research({
                'query': query,
                'sources': sources,
                'max_results': 5
            })
            trend_results.append({
                'query': query,
                'results': query_results
            })
        
        # Analyze trends
        trend_analysis = await self._analyze_trends(topic, trend_results, time_period)
        
        return {
            'research_type': 'trend_analysis',
            'topic': topic,
            'time_period': time_period,
            'trend_research': trend_results,
            'trend_analysis': trend_analysis
        }

    async def _literature_review_research(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct literature review style research"""
        topic = input_data.get('topic', '')
        focus_areas = input_data.get('focus_areas', [])
        sources = input_data.get('sources', ['documents', 'privateGPT'])
        
        if not topic:
            raise ValueError("Topic is required for literature review")
        
        # Research topic and focus areas
        literature_results = []
        
        # Main topic research
        main_research = await self._conduct_query_research({
            'query': topic,
            'sources': sources,
            'max_results': 10
        })
        literature_results.append({
            'area': 'main_topic',
            'query': topic,
            'results': main_research
        })
        
        # Focus area research
        for area in focus_areas:
            area_query = f"{topic} {area}"
            area_research = await self._conduct_query_research({
                'query': area_query,
                'sources': sources,
                'max_results': 5
            })
            literature_results.append({
                'area': area,
                'query': area_query,
                'results': area_research
            })
        
        # Generate literature review
        review = await self._generate_literature_review(topic, focus_areas, literature_results)
        
        return {
            'research_type': 'literature_review',
            'topic': topic,
            'focus_areas': focus_areas,
            'literature_research': literature_results,
            'literature_review': review
        }

    async def _search_documents(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search through document sources"""
        results = []
        
        # Search privateGPT documents
        try:
            privateGPT_results = await self.query_privateGPT(query)
            results.append({
                'source': 'privateGPT',
                'content': privateGPT_results,
                'relevance': 'high'
            })
        except Exception as e:
            self.logger.warning(f"Error querying privateGPT: {e}")
        
        # Search text vault
        try:
            text_vault_results = await self._search_text_vault(query, max_results)
            results.extend(text_vault_results)
        except Exception as e:
            self.logger.warning(f"Error searching text vault: {e}")
        
        return results[:max_results]

    async def _search_knowledge_base(self, query: str, max_results: int = 10) -> str:
        """Search internal knowledge base using AI model"""
        prompt = f"""
Based on your knowledge, provide comprehensive information about: {query}

Please include:
1. Key concepts and definitions
2. Important facts and figures
3. Current understanding and consensus
4. Notable developments or changes
5. Practical applications or implications

Research response:
"""
        
        response = await self.call_lm_studio(prompt, max_tokens=600)
        return response

    async def _query_privateGPT_research(self, query: str, max_results: int = 10) -> str:
        """Query privateGPT specifically for research"""
        research_query = f"Research and provide detailed information about: {query}"
        return await self.query_privateGPT(research_query)

    async def _search_text_vault(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search through TEXT-VAULT documents"""
        results = []
        
        try:
            # Get list of documents in TEXT-VAULT
            documents = self.list_documents(str(self.text_vault_path))
            
            # Search through documents
            for doc_path in documents[:20]:  # Limit for performance
                try:
                    content = self.read_document(doc_path)
                    if query.lower() in content.lower():
                        # Extract relevant excerpts
                        excerpts = self._extract_relevant_excerpts(content, query)
                        results.append({
                            'source': 'text_vault',
                            'document': doc_path,
                            'excerpts': excerpts,
                            'relevance': 'medium'
                        })
                        
                        if len(results) >= max_results:
                            break
                            
                except Exception as e:
                    self.logger.debug(f"Error reading document {doc_path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error searching text vault: {e}")
        
        return results

    async def _search_prompts(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search through AI prompts"""
        results = []
        
        try:
            # Get list of prompt files
            prompt_files = self.list_documents(str(self.aiprompts_path))
            
            # Search through prompt files
            for prompt_path in prompt_files[:10]:
                try:
                    content = self.read_document(prompt_path)
                    if query.lower() in content.lower():
                        results.append({
                            'source': 'prompts',
                            'file': prompt_path,
                            'content': content[:500] + "..." if len(content) > 500 else content,
                            'relevance': 'medium'
                        })
                        
                        if len(results) >= max_results:
                            break
                            
                except Exception as e:
                    self.logger.debug(f"Error reading prompt file {prompt_path}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error searching prompts: {e}")
        
        return results

    def _extract_relevant_excerpts(self, content: str, query: str, context_chars: int = 300) -> List[str]:
        """Extract relevant excerpts from content"""
        excerpts = []
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find all occurrences of query terms
        start = 0
        while True:
            pos = content_lower.find(query_lower, start)
            if pos == -1:
                break
            
            # Extract context around the match
            excerpt_start = max(0, pos - context_chars // 2)
            excerpt_end = min(len(content), pos + len(query) + context_chars // 2)
            excerpt = content[excerpt_start:excerpt_end].strip()
            
            if excerpt not in excerpts:
                excerpts.append(excerpt)
            
            start = pos + 1
            
            # Limit number of excerpts
            if len(excerpts) >= 3:
                break
        
        return excerpts

    async def _synthesize_research_results(self, query: str, research_results: List[Dict[str, Any]]) -> str:
        """Synthesize research results into a coherent summary"""
        
        # Prepare synthesis prompt
        results_text = ""
        for result in research_results:
            source = result['source']
            results_content = result['results']
            
            if isinstance(results_content, str):
                results_text += f"\n{source.upper()} RESULTS:\n{results_content}\n"
            elif isinstance(results_content, list):
                for item in results_content:
                    if isinstance(item, dict):
                        results_text += f"\n{source.upper()}:\n{json.dumps(item, indent=2)}\n"
                    else:
                        results_text += f"\n{source.upper()}:\n{item}\n"
        
        prompt = f"""
Synthesize the following research results for the query: "{query}"

Research Results:
{results_text[:2000]}...

Please provide:
1. Key findings summary
2. Common themes across sources
3. Important insights
4. Gaps or contradictions (if any)
5. Conclusion

Synthesis:
"""
        
        synthesis = await self.call_lm_studio(prompt, max_tokens=500)
        return synthesis

    async def _generate_comparison(self, topics: List[str], topic_results: Dict[str, Any], 
                                  aspects: List[str]) -> str:
        """Generate comparison between topics"""
        
        # Prepare comparison data
        comparison_data = ""
        for topic, results in topic_results.items():
            comparison_data += f"\n{topic.upper()}:\n"
            if 'synthesis' in results:
                comparison_data += f"{results['synthesis']}\n"
        
        prompt = f"""
Compare the following topics: {', '.join(topics)}

Focus on these aspects: {', '.join(aspects)}

Research data:
{comparison_data[:1500]}...

Please provide:
1. Similarities between topics
2. Key differences
3. Strengths and weaknesses of each
4. Comparative analysis for each aspect
5. Overall comparison summary

Comparison:
"""
        
        comparison = await self.call_lm_studio(prompt, max_tokens=600)
        return comparison

    async def _generate_comprehensive_report(self, topic: str, comprehensive_results: List[Dict[str, Any]], 
                                           depth: str) -> str:
        """Generate comprehensive research report"""
        
        # Prepare report data
        report_data = ""
        for result in comprehensive_results:
            query = result['query']
            results = result['results']
            report_data += f"\n{query.upper()}:\n"
            if 'synthesis' in results:
                report_data += f"{results['synthesis']}\n"
        
        prompt = f"""
Generate a comprehensive {depth} research report on: {topic}

Based on the following research:
{report_data[:2000]}...

Please structure the report with:
1. Executive Summary
2. Introduction/Background
3. Key Findings
4. Detailed Analysis
5. Implications and Applications
6. Conclusions and Recommendations

Comprehensive Report:
"""
        
        report = await self.call_lm_studio(prompt, max_tokens=800)
        return report

    async def _analyze_claim(self, claim: str, research: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a claim based on research"""
        
        # Prepare analysis prompt
        research_text = ""
        if 'synthesis' in research:
            research_text = research['synthesis']
        
        prompt = f"""
Fact-check this claim: "{claim}"

Based on this research:
{research_text[:1000]}...

Please provide:
1. Verification status (True/False/Partially True/Unverified)
2. Supporting evidence
3. Contradicting evidence (if any)
4. Confidence level (High/Medium/Low)
5. Additional context needed

Analysis:
"""
        
        analysis_text = await self.call_lm_studio(prompt, max_tokens=400)
        
        # Parse analysis for structured response
        return {
            'claim': claim,
            'analysis': analysis_text,
            'research_basis': research_text[:200] + "..." if len(research_text) > 200 else research_text
        }

    async def _analyze_trends(self, topic: str, trend_results: List[Dict[str, Any]], 
                            time_period: str) -> str:
        """Analyze trends based on research"""
        
        # Prepare trend data
        trend_data = ""
        for result in trend_results:
            query = result['query']
            results = result['results']
            trend_data += f"\n{query}:\n"
            if 'synthesis' in results:
                trend_data += f"{results['synthesis']}\n"
        
        prompt = f"""
Analyze trends for: {topic} over {time_period}

Based on this research:
{trend_data[:1500]}...

Please provide:
1. Identified trends and patterns
2. Trend direction (growing/declining/stable)
3. Key drivers of change
4. Future predictions
5. Implications and recommendations

Trend Analysis:
"""
        
        analysis = await self.call_lm_studio(prompt, max_tokens=600)
        return analysis

    async def _generate_literature_review(self, topic: str, focus_areas: List[str], 
                                        literature_results: List[Dict[str, Any]]) -> str:
        """Generate literature review"""
        
        # Prepare literature data
        literature_data = ""
        for result in literature_results:
            area = result['area']
            results = result['results']
            literature_data += f"\n{area.upper()}:\n"
            if 'synthesis' in results:
                literature_data += f"{results['synthesis']}\n"
        
        prompt = f"""
Generate a literature review on: {topic}

Focus areas: {', '.join(focus_areas)}

Based on this research:
{literature_data[:2000]}...

Please structure as:
1. Introduction and scope
2. Literature overview by focus area
3. Key themes and findings
4. Gaps in current knowledge
5. Future research directions
6. Conclusions

Literature Review:
"""
        
        review = await self.call_lm_studio(prompt, max_tokens=800)
        return review