#!/usr/bin/env python3
"""
Smart Caching System for AI Agent Ecosystem
Implements intelligent result caching with TTL, LRU eviction, and cache warming
"""

import asyncio
import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import pickle
import gzip

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SmartCache:
    """Smart caching system with TTL, LRU eviction, and analytics"""
    
    def __init__(self, db_path: Path, config: Dict[str, Any]):
        self.db_path = db_path
        self.config = config
        self.max_memory_mb = config.get('max_memory_mb', 512)
        self.default_ttl = config.get('default_ttl_seconds', 3600)  # 1 hour
        self.cleanup_interval = config.get('cleanup_interval', 300)  # 5 minutes
        self.compression_enabled = config.get('compression', True)
        
        # In-memory cache for frequently accessed items
        self.memory_cache = {}
        self.memory_usage_bytes = 0
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_hits': 0,
            'disk_hits': 0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.warming_task = None
        
    async def start(self):
        """Start caching system"""
        logger.info("Starting Smart Caching System")
        
        # Initialize database
        await self._init_database()
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_expired())
        self.warming_task = asyncio.create_task(self._cache_warming())
        
    async def stop(self):
        """Stop caching system"""
        logger.info("Stopping Smart Caching System")
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.warming_task:
            self.warming_task.cancel()
            
        # Flush memory cache to disk
        await self._flush_memory_cache()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry.expires_at > datetime.now():
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    return entry.value
                else:
                    # Expired, remove from memory
                    del self.memory_cache[key]
                    self.memory_usage_bytes -= entry.size_bytes
            
            # Check disk cache
            entry = await self._get_from_disk(key)
            if entry and entry.expires_at > datetime.now():
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Update database
                await self._update_entry_stats(entry)
                
                # Add to memory cache if frequently accessed
                if entry.access_count > 2:
                    await self._add_to_memory_cache(entry)
                
                self.stats['hits'] += 1
                self.stats['disk_hits'] += 1
                return entry.value
            
            # Cache miss
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, 
                 ttl_seconds: int = None, 
                 tags: List[str] = None,
                 metadata: Dict[str, Any] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl_seconds or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Serialize and compress value
            serialized_value = pickle.dumps(value)
            if self.compression_enabled:
                serialized_value = gzip.compress(serialized_value)
            
            size_bytes = len(serialized_value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size_bytes,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Store in database
            await self._store_to_disk(entry, serialized_value)
            
            # Add to memory cache if it fits
            if size_bytes < self.max_memory_mb * 1024 * 1024 * 0.1:  # Max 10% of cache for single item
                await self._add_to_memory_cache(entry)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                del self.memory_cache[key]
                self.memory_usage_bytes -= entry.size_bytes
            
            # Remove from disk cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM result_cache WHERE cache_key = ?', (key,))
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all cache entries with specified tags"""
        try:
            deleted_count = 0
            
            # Get entries with matching tags
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for tag in tags:
                cursor.execute('''
                    SELECT cache_key FROM result_cache 
                    WHERE json_extract(metadata, '$.tags') LIKE ?
                ''', (f'%"{tag}"%',))
                
                keys_to_delete = [row[0] for row in cursor.fetchall()]
                
                for key in keys_to_delete:
                    await self.delete(key)
                    deleted_count += 1
            
            conn.close()
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache delete by tags error: {e}")
            return 0
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            self.memory_usage_bytes = 0
            
            # Clear disk cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM result_cache')
            conn.commit()
            conn.close()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    async def get_or_set(self, key: str, 
                        factory_func: Callable, 
                        ttl_seconds: int = None,
                        tags: List[str] = None) -> Any:
        """Get value from cache or compute and store it"""
        try:
            # Try to get from cache first
            value = await self.get(key)
            if value is not None:
                return value
            
            # Compute value
            if asyncio.iscoroutinefunction(factory_func):
                value = await factory_func()
            else:
                value = factory_func()
            
            # Store in cache
            await self.set(key, value, ttl_seconds, tags)
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get_or_set error for key {key}: {e}")
            # Return computed value even if caching fails
            if asyncio.iscoroutinefunction(factory_func):
                return await factory_func()
            else:
                return factory_func()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute('SELECT COUNT(*) FROM result_cache')
            total_entries = cursor.fetchone()[0]
            
            # Expired entries
            cursor.execute('SELECT COUNT(*) FROM result_cache WHERE expires_at < ?', 
                          (datetime.now().isoformat(),))
            expired_entries = cursor.fetchone()[0]
            
            # Total size
            cursor.execute('SELECT SUM(LENGTH(result)) FROM result_cache')
            total_size_bytes = cursor.fetchone()[0] or 0
            
            conn.close()
            
            hit_rate = self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)
            
            return {
                'total_entries': total_entries,
                'expired_entries': expired_entries,
                'memory_entries': len(self.memory_cache),
                'memory_usage_mb': self.memory_usage_bytes / (1024 * 1024),
                'disk_usage_mb': total_size_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'stats': self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {'error': str(e)}
    
    async def _init_database(self):
        """Initialize cache database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure result_cache table exists with proper schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS result_cache (
                    cache_key TEXT PRIMARY KEY,
                    result BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON result_cache(expires_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON result_cache(last_accessed)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
    
    async def _get_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Get entry from disk cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT cache_key, result, created_at, expires_at, access_count, 
                       last_accessed, metadata 
                FROM result_cache WHERE cache_key = ?
            ''', (key,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                # Deserialize value
                serialized_value = row[1]
                if self.compression_enabled:
                    try:
                        serialized_value = gzip.decompress(serialized_value)
                    except:
                        pass  # Not compressed
                
                value = pickle.loads(serialized_value)
                
                metadata = json.loads(row[6]) if row[6] else {}
                
                return CacheEntry(
                    key=row[0],
                    value=value,
                    created_at=datetime.fromisoformat(row[2]),
                    expires_at=datetime.fromisoformat(row[3]),
                    access_count=row[4],
                    last_accessed=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                    size_bytes=len(row[1]),
                    tags=metadata.get('tags', []),
                    metadata=metadata
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting from disk cache: {e}")
            return None
    
    async def _store_to_disk(self, entry: CacheEntry, serialized_value: bytes):
        """Store entry to disk cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metadata = {
                'tags': entry.tags,
                **entry.metadata
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO result_cache 
                (cache_key, result, created_at, expires_at, access_count, 
                 last_accessed, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.key,
                serialized_value,
                entry.created_at.isoformat(),
                entry.expires_at.isoformat(),
                entry.access_count,
                entry.last_accessed.isoformat(),
                json.dumps(metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing to disk cache: {e}")
    
    async def _add_to_memory_cache(self, entry: CacheEntry):
        """Add entry to memory cache with LRU eviction"""
        try:
            # Check if we need to evict entries
            while (self.memory_usage_bytes + entry.size_bytes > 
                   self.max_memory_mb * 1024 * 1024):
                await self._evict_lru_entry()
            
            self.memory_cache[entry.key] = entry
            self.memory_usage_bytes += entry.size_bytes
            
        except Exception as e:
            logger.error(f"Error adding to memory cache: {e}")
    
    async def _evict_lru_entry(self):
        """Evict least recently used entry from memory cache"""
        if not self.memory_cache:
            return
        
        # Find LRU entry
        lru_key = min(self.memory_cache.keys(), 
                     key=lambda k: self.memory_cache[k].last_accessed)
        
        entry = self.memory_cache[lru_key]
        del self.memory_cache[lru_key]
        self.memory_usage_bytes -= entry.size_bytes
        self.stats['evictions'] += 1
    
    async def _update_entry_stats(self, entry: CacheEntry):
        """Update entry statistics in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE result_cache 
                SET access_count = ?, last_accessed = ?
                WHERE cache_key = ?
            ''', (entry.access_count, entry.last_accessed.isoformat(), entry.key))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating entry stats: {e}")
    
    async def _cleanup_expired(self):
        """Cleanup expired cache entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = datetime.now()
                
                # Remove expired entries from memory cache
                expired_keys = [
                    key for key, entry in self.memory_cache.items()
                    if entry.expires_at <= current_time
                ]
                
                for key in expired_keys:
                    entry = self.memory_cache[key]
                    del self.memory_cache[key]
                    self.memory_usage_bytes -= entry.size_bytes
                
                # Remove expired entries from disk cache
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM result_cache WHERE expires_at < ?', 
                              (current_time.isoformat(),))
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                if deleted_count > 0 or expired_keys:
                    logger.info(f"Cleaned up {deleted_count + len(expired_keys)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _cache_warming(self):
        """Cache warming for frequently accessed entries"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Get frequently accessed entries that are about to expire
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                near_expiry = datetime.now() + timedelta(minutes=30)
                cursor.execute('''
                    SELECT cache_key, access_count 
                    FROM result_cache 
                    WHERE expires_at < ? AND access_count > 5
                    ORDER BY access_count DESC
                    LIMIT 10
                ''', (near_expiry.isoformat(),))
                
                warming_candidates = cursor.fetchall()
                conn.close()
                
                if warming_candidates:
                    logger.info(f"Found {len(warming_candidates)} cache entries for warming")
                    # Here you could implement logic to refresh these entries
                    # by calling their original factory functions
                
            except Exception as e:
                logger.error(f"Error in cache warming: {e}")
    
    async def _flush_memory_cache(self):
        """Flush memory cache to disk"""
        try:
            for entry in self.memory_cache.values():
                # Update access stats in database
                await self._update_entry_stats(entry)
            
            logger.info(f"Flushed {len(self.memory_cache)} entries from memory cache")
            
        except Exception as e:
            logger.error(f"Error flushing memory cache: {e}")

def cache_result(cache: SmartCache, ttl_seconds: int = None, tags: List[str] = None):
    """Decorator for caching function results"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = cache.generate_key(key_data)
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl_seconds, tags)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we need to handle async cache operations
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator