/**
 * AI Agent Ecosystem Dashboard JavaScript
 * Handles navigation, real-time updates, and API interactions
 */

class AIEcosystemDashboard {
    constructor() {
        this.apiBaseUrl = window.location.hostname === 'localhost' ? 
            'http://localhost:8000' : 
            `http://${window.location.hostname}:8000`;
        
        this.websocket = null;
        this.updateInterval = null;
        
        // Enhanced state management
        this.lastGPUData = {};
        this.lastModelsData = {};
        this.isLoadingGPUData = false;
        this.gpuUpdateDebounce = 0;
        
        this.init();
    }

    init() {
        console.log('Initializing AI Ecosystem Dashboard...');
        this.setupNavigation();
        this.setupSidebar();
        this.setupWebSocket();
        
        // Add a delay to ensure DOM is fully rendered
        setTimeout(() => {
            console.log('Setting up research categories...');
            this.setupResearchCategories();
        }, 500);
        
        this.startPeriodicUpdates();
        this.loadInitialData();
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const sections = document.querySelectorAll('.content-section');
        const pageTitle = document.querySelector('.page-title');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                const targetSection = link.getAttribute('data-section');
                
                // Get current section for cleanup
                const currentSection = document.querySelector('.content-section.active');
                const currentSectionId = currentSection ? currentSection.id : null;
                
                // Unsubscribe from previous section's updates
                if (currentSectionId === 'gpu') {
                    this.unsubscribeFromUpdates('gpu_metrics');
                }
                
                // Update active nav item
                document.querySelector('.nav-item.active')?.classList.remove('active');
                link.parentElement.classList.add('active');
                
                // Update active section
                sections.forEach(section => section.classList.remove('active'));
                document.getElementById(targetSection)?.classList.add('active');
                
                // Subscribe to new section's updates
                if (targetSection === 'gpu') {
                    this.subscribeToUpdates('gpu_metrics');
                }
                
                // Update page title
                const sectionTitle = link.querySelector('span').textContent;
                pageTitle.textContent = sectionTitle;
                
                // Load section-specific data
                this.loadSectionData(targetSection);
            });
        });
    }

    setupSidebar() {
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        
        if (sidebarToggle) {
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
        }

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
                    sidebar.classList.remove('open');
                }
            }
        });
    }

    setupWebSocket() {
        try {
            const wsUrl = this.apiBaseUrl.replace('http', 'ws') + '/ws';
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateSystemStatus('online');
                // Subscribe to basic system metrics
                this.subscribeToUpdates('system_metrics');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateSystemStatus('offline');
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    this.setupWebSocket();
                }, 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateSystemStatus('offline');
            };
        } catch (error) {
            console.error('Failed to setup WebSocket:', error);
            this.updateSystemStatus('offline');
        }
    }
    
    subscribeToUpdates(subscription) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'subscribe',
                subscription: subscription
            }));
            console.log(`Subscribed to ${subscription}`);
        }
    }
    
    unsubscribeFromUpdates(subscription) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'unsubscribe',
                subscription: subscription
            }));
            console.log(`Unsubscribed from ${subscription}`);
        }
    }

    handleWebSocketMessage(data) {
        console.log('Status update:', data);
        
        switch (data.type) {
            case 'agent_status':
                this.updateAgentStatus(data.payload);
                break;
            case 'task_update':
                this.updateTaskStatus(data.payload);
                break;
            case 'gpu_metrics':
                this.updateRealtimeGPUMetrics(data.metrics);
                break;
            case 'system_metrics':
                console.log('System metrics update:', data);
                this.updateSystemMetrics(data.metrics);
                this.updateRecentEvents(data);
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    startPeriodicUpdates() {
        // Update dashboard data every 30 seconds
        this.updateInterval = setInterval(() => {
            this.loadDashboardData();
        }, 30000);
    }

    async loadInitialData() {
        await this.loadDashboardData();
    }

    async loadDashboardData() {
        try {
            const [healthData, metricsData, tasksData] = await Promise.all([
                this.fetchAPI('/health'),
                this.fetchAPI('/metrics'),
                this.fetchAPI('/tasks')
            ]);

            this.updateSystemStatus(healthData.status === 'healthy' ? 'online' : 'offline');
            this.updateSystemOverview(metricsData);
            this.updateAgentStatuses(metricsData);
            this.updateRecentTasks(tasksData);
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.updateSystemStatus('offline');
        }
    }

    async loadSectionData(section) {
        switch (section) {
            case 'agents':
                await this.loadAgentsData();
                break;
            case 'tasks':
                await this.loadTasksData();
                break;
            case 'gpu':
                await this.loadGPUData();
                break;
            // Add more sections as needed
        }
    }

    async fetchAPI(endpoint) {
        const response = await fetch(`${this.apiBaseUrl}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        return await response.json();
    }

    updateSystemStatus(status) {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.system-status span:last-child');
        
        if (statusIndicator && statusText) {
            statusIndicator.className = `status-indicator ${status}`;
            statusText.textContent = status === 'online' ? 'System Online' : 'System Offline';
        }
    }

    updateSystemOverview(data) {
        // Update overview statistics
        const elements = {
            'active-agents': data?.active_agents || 5,
            'pending-tasks': data?.pending_tasks || 0,
            'completed-today': data?.completed_today || 0,
            'gpu-usage': data?.gpu_usage ? `${data.gpu_usage}%` : '0%'
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    updateAgentStatuses(metricsData) {
        const agentElements = document.querySelectorAll('.agent-status');
        
        agentElements.forEach(element => {
            const agentType = element.getAttribute('data-agent');
            const stateElement = element.querySelector('.agent-state');
            const metricElement = element.querySelector('.metric');
            
            if (metricsData && metricsData.agent_instances) {
                // Count instances by type and get status
                const instances = Object.values(metricsData.agent_instances).filter(
                    instance => instance.agent_type === agentType
                );
                
                if (instances.length > 0) {
                    const activeCount = instances.filter(i => i.status !== 'idle').length;
                    const allIdle = instances.every(i => i.status === 'idle');
                    
                    stateElement.textContent = allIdle ? 'Idle' : 'Active';
                    stateElement.className = `agent-state ${allIdle ? 'idle' : 'active'}`;
                    
                    if (metricElement) {
                        metricElement.textContent = `${instances.length} instance${instances.length !== 1 ? 's' : ''}`;
                    }
                } else {
                    stateElement.textContent = 'Offline';
                    stateElement.className = 'agent-state offline';
                    if (metricElement) {
                        metricElement.textContent = '0 instances';
                    }
                }
            }
        });
    }

    updateAgentStatus(agentData) {
        const agentElement = document.querySelector(`[data-agent="${agentData.type}"]`);
        if (agentElement) {
            const stateElement = agentElement.querySelector('.agent-state');
            const metricElement = agentElement.querySelector('.metric');
            
            if (stateElement) {
                stateElement.textContent = agentData.status;
                stateElement.className = `agent-state ${agentData.status.toLowerCase()}`;
            }
            
            if (metricElement) {
                metricElement.textContent = `${agentData.active_tasks} active`;
            }
        }
    }

    updateTaskStatus(taskData) {
        // Update task in the recent tasks list
        const taskList = document.querySelector('.task-list');
        if (taskList) {
            // This would be more sophisticated in a real implementation
            console.log('Task update:', taskData);
        }
    }

    updateGPUMetrics(gpuData) {
        gpuData.forEach((gpu, index) => {
            const gpuCard = document.querySelectorAll('.gpu-card')[index];
            if (gpuCard) {
                // Update temperature
                const tempElement = gpuCard.querySelector('.gpu-temp');
                if (tempElement) {
                    tempElement.textContent = `${gpu.temperature}°C`;
                }
                
                // Update VRAM usage
                const vramBar = gpuCard.querySelector('.progress-fill');
                const vramText = gpuCard.querySelector('.metric-bar span');
                if (vramBar && vramText) {
                    const vramPercent = (gpu.vram_used / gpu.vram_total) * 100;
                    vramBar.style.width = `${vramPercent}%`;
                    vramText.textContent = `${gpu.vram_used}GB / ${gpu.vram_total}GB`;
                }
                
                // Update GPU usage
                const gpuBars = gpuCard.querySelectorAll('.progress-fill');
                const gpuTexts = gpuCard.querySelectorAll('.metric-bar span');
                if (gpuBars[1] && gpuTexts[1]) {
                    gpuBars[1].style.width = `${gpu.usage}%`;
                    gpuTexts[1].textContent = `${gpu.usage}%`;
                }
                
                // Update model info
                const modelElement = gpuCard.querySelector('.gpu-model span');
                if (modelElement) {
                    modelElement.textContent = `Running: ${gpu.current_model || 'Idle'}`;
                }
            }
        });
    }

    updateRecentTasks(tasksData) {
        const taskList = document.querySelector('.task-list');
        if (!taskList || !tasksData || !tasksData.tasks) return;
        
        // Clear existing tasks
        taskList.innerHTML = '';
        
        // Add recent tasks (limit to 3 for dashboard)
        const recentTasks = tasksData.tasks.slice(0, 3);
        recentTasks.forEach(task => {
            const taskElement = this.createTaskElement(task);
            taskList.appendChild(taskElement);
        });
    }

    createTaskElement(task) {
        const taskItem = document.createElement('div');
        taskItem.className = `task-item ${task.status}`;
        
        const iconMap = {
            'document_analysis': 'fas fa-file-alt',
            'code_generation': 'fas fa-code',
            'research': 'fas fa-search',
            'data_processing': 'fas fa-database',
            'task_coordination': 'fas fa-sitemap'
        };
        
        taskItem.innerHTML = `
            <div class="task-icon">
                <i class="${iconMap[task.type] || 'fas fa-tasks'}"></i>
            </div>
            <div class="task-info">
                <h4>${task.title || task.type}</h4>
                <span class="task-time">${this.formatTaskTime(task.created_at)}</span>
            </div>
            <div class="task-status">
                <span class="status ${task.status}">${task.status}</span>
            </div>
        `;
        
        return taskItem;
    }

    formatTaskTime(timestamp) {
        if (!timestamp) return 'Unknown';
        
        const now = new Date();
        const taskTime = new Date(timestamp);
        const diffMs = now - taskTime;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} minutes ago`;
        
        const diffHours = Math.floor(diffMins / 60);
        if (diffHours < 24) return `${diffHours} hours ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays} days ago`;
    }

    async loadSectionData(sectionName) {
        switch (sectionName) {
            case 'dashboard':
                await this.loadDashboardData();
                break;
            case 'agents':
                await this.loadAgentsData();
                break;
            case 'tasks':
                await this.loadTasksData();
                break;
            case 'documents':
                await this.loadDocumentsData();
                break;
            case 'code':
                await this.loadCodeData();
                break;
            case 'research':
                await this.loadResearchData();
                break;
            case 'backends':
                await this.loadBackendsData();
                break;
            case 'gpu':
                await this.loadGPUData();
                break;
            case 'settings':
                await this.loadSettingsData();
                break;
            default:
                console.log(`No data loader for section: ${sectionName}`);
        }
    }

    // Placeholder methods for future sections
    async loadAgentsData() {
        console.log('Loading agents data...');
    }

    async loadTasksData() {
        console.log('Loading tasks data...');
    }

    async loadDocumentsData() {
        console.log('Loading documents data...');
    }

    async loadCodeData() {
        console.log('Loading code data...');
    }

    async loadGPUData() {
        console.log('Loading GPU data...');
        
        // Only load GPU data if we're on the GPU page to avoid unnecessary API calls
        const currentSection = document.querySelector('.content-section.active');
        if (!currentSection || currentSection.id !== 'gpu') {
            return;
        }
        
        if (this.isLoadingGPUData) {
            return; // Prevent concurrent loading
        }
        
        this.isLoadingGPUData = true;
        
        try {
            const gpuData = await this.fetchAPI('/gpu-platform/metrics');
            this.updateGPUDisplay(gpuData);
            this.lastGPUData = gpuData;
            this.initializeGPUControls();
        } catch (error) {
            console.error('Failed to load GPU data:', error);
            this.displayGPUError();
        } finally {
            this.isLoadingGPUData = false;
        }
    }
    
    initializeGPUControls() {
        // Initialize refresh button
        const refreshBtn = document.getElementById('refresh-gpu-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadGPUData();
            });
        }
        
        // Initialize model management
        const loadModelBtn = document.getElementById('load-model-btn');
        const unloadModelBtn = document.getElementById('unload-model-btn');
        
        if (loadModelBtn) {
            loadModelBtn.addEventListener('click', () => {
                this.loadModelOnGPU();
            });
        }
        
        if (unloadModelBtn) {
            unloadModelBtn.addEventListener('click', () => {
                this.unloadModelFromGPU();
            });
        }
        
        // Populate GPU selector
        this.updateGPUSelector();
    }
    
    updateGPUSelector() {
        const gpuSelect = document.getElementById('gpu-select');
        if (gpuSelect && this.lastGPUData && this.lastGPUData.gpus) {
            gpuSelect.innerHTML = '<option value="">Auto-select best GPU</option>';
            Object.entries(this.lastGPUData.gpus).forEach(([index, gpu]) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `GPU ${index}: ${gpu.name}`;
                gpuSelect.appendChild(option);
            });
        }
    }
    
    async loadModelOnGPU() {
        const modelSelect = document.getElementById('model-select');
        const gpuSelect = document.getElementById('gpu-select');
        
        const modelName = modelSelect.value;
        const gpuIndex = gpuSelect.value || null;
        
        if (!modelName) {
            alert('Please select a model to load');
            return;
        }
        
        try {
            const response = await this.fetchAPI('/gpu-platform/models/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: modelName,
                    gpu_index: gpuIndex ? parseInt(gpuIndex) : null,
                    memory_mb: 4000 // Default memory requirement
                })
            });
            
            if (response.success) {
                alert(`Model ${modelName} loaded successfully on GPU ${response.gpu_index}`);
                this.loadGPUData(); // Refresh display
            }
        } catch (error) {
            console.error('Failed to load model:', error);
            alert('Failed to load model');
        }
    }
    
    async unloadModelFromGPU() {
        const modelSelect = document.getElementById('model-select');
        const gpuSelect = document.getElementById('gpu-select');
        
        const modelName = modelSelect.value;
        const gpuIndex = gpuSelect.value || null;
        
        if (!modelName) {
            alert('Please select a model to unload');
            return;
        }
        
        try {
            const response = await this.fetchAPI('/gpu-platform/models/unload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_name: modelName,
                    gpu_index: gpuIndex ? parseInt(gpuIndex) : null
                })
            });
            
            if (response.success) {
                alert(`Model ${modelName} unloaded successfully`);
                this.loadGPUData(); // Refresh display
            }
        } catch (error) {
            console.error('Failed to unload model:', error);
            alert('Failed to unload model');
        }
    }
    
    updateGPUDisplay(gpuData) {
        const gpuContainer = document.querySelector('.gpu-metrics');
        if (!gpuContainer) return;
        
        if (!gpuData || !gpuData.gpus || Object.keys(gpuData.gpus).length === 0) {
            gpuContainer.innerHTML = '<div class="empty-state"><i class="fas fa-microchip"></i><p>No GPUs detected</p></div>';
            return;
        }
        
        // Update GPU cards
        let gpuCardsHTML = '';
        Object.entries(gpuData.gpus).forEach(([index, gpu]) => {
            gpuCardsHTML += `
                <div class="gpu-card">
                    <div class="gpu-header">
                        <h4>GPU ${index}: ${gpu.name}</h4>
                        <div class="gpu-status ${gpu.is_available ? 'online' : 'offline'}">
                            ${gpu.is_available ? 'Available' : 'Busy'}
                        </div>
                    </div>
                    <div class="gpu-metrics-grid">
                        <div class="metric-item">
                            <label>Memory Usage</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${gpu.memory_percent}%"></div>
                            </div>
                            <span>${gpu.memory_used}MB / ${gpu.memory_total}MB (${gpu.memory_percent.toFixed(1)}%)</span>
                        </div>
                        <div class="metric-item">
                            <label>GPU Utilization</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${gpu.utilization}%"></div>
                            </div>
                            <span>${gpu.utilization.toFixed(1)}%</span>
                        </div>
                        <div class="metric-item">
                            <label>Temperature</label>
                            <div class="temperature-display">
                                <span class="temperature-value">${gpu.temperature}°C</span>
                            </div>
                        </div>
                        <div class="metric-item">
                            <label>Power Usage</label>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${gpu.power_percent}%"></div>
                            </div>
                            <span>${gpu.power_usage.toFixed(1)}W / ${gpu.power_limit.toFixed(1)}W</span>
                        </div>
                        ${gpu.current_model ? `
                            <div class="metric-item">
                                <label>Current Model</label>
                                <span class="model-name">${gpu.current_model}</span>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });
        
        gpuContainer.innerHTML = gpuCardsHTML;
        
        // Update summary stats
        this.updateGPUSummary(gpuData.summary);
    }
    
    updateGPUSummary(summary) {
        const elements = {
            'total-gpus': summary.total_gpus || 0,
            'available-gpus': summary.available_gpus || 0,
            'total-memory': `${(summary.total_memory / 1024).toFixed(1)}GB`,
            'used-memory': `${(summary.used_memory / 1024).toFixed(1)}GB`,
            'avg-utilization': `${summary.average_utilization.toFixed(1)}%`,
            'avg-temperature': `${summary.average_temperature.toFixed(1)}°C`
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }
    
    updateRealtimeGPUMetrics(gpuData) {
        // Only update if we're on the GPU page
        const currentSection = document.querySelector('.content-section.active');
        if (!currentSection || currentSection.id !== 'gpu') {
            return;
        }
        
        // Debounce updates to prevent flashing
        const currentTime = Date.now();
        if (currentTime - this.gpu_update_debounce < 1000) { // 1 second debounce
            return;
        }
        this.gpu_update_debounce = currentTime;
        
        this.updateGPUDisplay(gpuData);
    }
    
    displayGPUError() {
        const gpuContainer = document.querySelector('.gpu-metrics');
        if (gpuContainer) {
            gpuContainer.innerHTML = '<div class="error-state"><i class="fas fa-exclamation-triangle"></i><p>Failed to load GPU data</p></div>';
        }
    }
    
    updateRecentEvents(data) {
        // Update recent events display with system metrics
        const eventsContainer = document.querySelector('.recent-events');
        if (eventsContainer && data.timestamp) {
            const eventTime = new Date(data.timestamp).toLocaleTimeString();
            const eventHTML = `
                <div class="event-item">
                    <div class="event-icon"><i class="fas fa-info-circle"></i></div>
                    <div class="event-content">
                        <div class="event-title">System Metrics Updated</div>
                        <div class="event-time">${eventTime}</div>
                    </div>
                </div>
            `;
            
            // Add to top of events list
            const existingEvents = eventsContainer.innerHTML;
            eventsContainer.innerHTML = eventHTML + existingEvents;
            
            // Keep only last 10 events
            const events = eventsContainer.querySelectorAll('.event-item');
            if (events.length > 10) {
                for (let i = 10; i < events.length; i++) {
                    events[i].remove();
                }
            }
        }
    }

    async loadSettingsData() {
        console.log('Loading settings data...');
        this.initializeSettingsInterface();
    }
    
    initializeSettingsInterface() {
        // Initialize profile selector
        const profileSelect = document.getElementById('current-profile');
        if (profileSelect) {
            profileSelect.addEventListener('change', (e) => {
                this.loadConfigurationProfile(e.target.value);
            });
        }
        
        // Initialize work mode radio buttons
        const workModeRadios = document.querySelectorAll('input[name="work-mode"]');
        workModeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.updateWorkModeConfiguration(e.target.value);
                }
            });
        });
        
        // Initialize GPU strategy radio buttons
        const gpuStrategyRadios = document.querySelectorAll('input[name="gpu-strategy"]');
        gpuStrategyRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.updateGPUStrategyConfiguration(e.target.value);
                }
            });
        });
        
        // Initialize range sliders with value updates
        this.initializeRangeSliders();
        
        // Initialize configuration buttons
        this.initializeConfigurationButtons();
        
        // Load current configuration
        this.loadCurrentConfiguration();
    }
    
    initializeRangeSliders() {
        const sliders = [
            { id: 'reserved-memory', valueId: 'reserved-memory-value', suffix: ' GB' },
            { id: 'memory-buffer', valueId: 'memory-buffer-value', suffix: ' GB' },
            { id: 'task-timeout', valueId: 'task-timeout-value', suffix: 's' },
            { id: 'cache-size', valueId: 'cache-size-value', suffix: ' GB' }
        ];
        
        sliders.forEach(slider => {
            const element = document.getElementById(slider.id);
            const valueElement = document.getElementById(slider.valueId);
            
            if (element && valueElement) {
                element.addEventListener('input', (e) => {
                    valueElement.textContent = e.target.value + slider.suffix;
                });
                
                // Set initial value
                valueElement.textContent = element.value + slider.suffix;
            }
        });
    }
    
    initializeConfigurationButtons() {
        // Apply Configuration
        const applyBtn = document.getElementById('apply-config-btn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                this.applyConfiguration();
            });
        }
        
        // Reset Configuration
        const resetBtn = document.getElementById('reset-config-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.resetConfiguration();
            });
        }
        
        // Preview Changes
        const previewBtn = document.getElementById('preview-config-btn');
        if (previewBtn) {
            previewBtn.addEventListener('click', () => {
                this.previewConfiguration();
            });
        }
        
        // Save Profile
        const saveProfileBtn = document.getElementById('save-profile-btn');
        if (saveProfileBtn) {
            saveProfileBtn.addEventListener('click', () => {
                this.saveConfigurationProfile();
            });
        }
        
        // Export Config
        const exportBtn = document.getElementById('export-config-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => {
                this.exportConfiguration();
            });
        }
    }
    
    async loadConfigurationProfile(profileName) {
        console.log('Loading configuration profile:', profileName);
        
        try {
            // This would typically load from an API endpoint
            const profiles = {
                development: {
                    workMode: 'development',
                    gpuStrategy: 'specialized',
                    primaryModel: 'codellama-34b',
                    fallbackModel: 'dolphin-8b',
                    agents: {
                        code_generator: { instances: 2, gpu: '0', autoscale: true, priority: '1' },
                        research_assistant: { instances: 1, gpu: 'both', autoscale: false, priority: '2' },
                        document_analyzer: { instances: 3, gpu: '1', autoscale: true, priority: '1' }
                    },
                    performance: {
                        reservedMemory: 1,
                        memoryBuffer: 0.5,
                        maxConcurrentTasks: 8,
                        taskTimeout: 300,
                        cacheSize: 4
                    }
                },
                research: {
                    workMode: 'research',
                    gpuStrategy: 'balanced',
                    primaryModel: 'llama-70b',
                    fallbackModel: 'dolphin-8b',
                    agents: {
                        code_generator: { instances: 1, gpu: '0', autoscale: false, priority: '3' },
                        research_assistant: { instances: 3, gpu: 'both', autoscale: true, priority: '1' },
                        document_analyzer: { instances: 5, gpu: 'both', autoscale: true, priority: '1' }
                    },
                    performance: {
                        reservedMemory: 2,
                        memoryBuffer: 1,
                        maxConcurrentTasks: 6,
                        taskTimeout: 600,
                        cacheSize: 8
                    }
                },
                production: {
                    workMode: 'production',
                    gpuStrategy: 'high_throughput',
                    primaryModel: 'llama-13b',
                    fallbackModel: 'mistral-7b',
                    agents: {
                        code_generator: { instances: 4, gpu: 'both', autoscale: true, priority: '2' },
                        research_assistant: { instances: 2, gpu: 'both', autoscale: true, priority: '2' },
                        document_analyzer: { instances: 6, gpu: 'both', autoscale: true, priority: '1' }
                    },
                    performance: {
                        reservedMemory: 0.5,
                        memoryBuffer: 0.25,
                        maxConcurrentTasks: 16,
                        taskTimeout: 120,
                        cacheSize: 2
                    }
                }
            };
            
            const profile = profiles[profileName];
            if (profile) {
                this.applyConfigurationProfile(profile);
            }
        } catch (error) {
            console.error('Failed to load configuration profile:', error);
        }
    }
    
    applyConfigurationProfile(profile) {
        // Update work mode
        const workModeRadio = document.querySelector(`input[name="work-mode"][value="${profile.workMode}"]`);
        if (workModeRadio) {
            workModeRadio.checked = true;
        }
        
        // Update GPU strategy
        const gpuStrategyRadio = document.querySelector(`input[name="gpu-strategy"][value="${profile.gpuStrategy}"]`);
        if (gpuStrategyRadio) {
            gpuStrategyRadio.checked = true;
        }
        
        // Update model selection
        const primaryModelSelect = document.getElementById('primary-model');
        const fallbackModelSelect = document.getElementById('fallback-model');
        if (primaryModelSelect) primaryModelSelect.value = profile.primaryModel;
        if (fallbackModelSelect) fallbackModelSelect.value = profile.fallbackModel;
        
        // Update agent configuration
        Object.entries(profile.agents).forEach(([agentType, config]) => {
            const instancesElement = document.getElementById(`${agentType}-instances`);
            const gpuSelect = document.getElementById(`${agentType}-gpu`);
            const autoscaleCheckbox = document.getElementById(`${agentType}-autoscale`);
            const prioritySelect = document.getElementById(`${agentType}-priority`);
            
            if (instancesElement) instancesElement.textContent = config.instances;
            if (gpuSelect) gpuSelect.value = config.gpu;
            if (autoscaleCheckbox) autoscaleCheckbox.checked = config.autoscale;
            if (prioritySelect) prioritySelect.value = config.priority;
        });
        
        // Update performance settings
        const performanceElements = {
            'reserved-memory': profile.performance.reservedMemory,
            'memory-buffer': profile.performance.memoryBuffer,
            'max-concurrent-tasks': profile.performance.maxConcurrentTasks,
            'task-timeout': profile.performance.taskTimeout,
            'cache-size': profile.performance.cacheSize
        };
        
        Object.entries(performanceElements).forEach(([elementId, value]) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.value = value;
                // Trigger input event to update display values
                element.dispatchEvent(new Event('input'));
            }
        });
    }
    
    updateWorkModeConfiguration(workMode) {
        console.log('Updating work mode configuration:', workMode);
        
        // Auto-update related settings based on work mode
        const configurations = {
            research: {
                gpuStrategy: 'balanced',
                primaryModel: 'llama-70b'
            },
            development: {
                gpuStrategy: 'specialized',
                primaryModel: 'codellama-34b'
            },
            production: {
                gpuStrategy: 'high_throughput',
                primaryModel: 'llama-13b'
            }
        };
        
        const config = configurations[workMode];
        if (config) {
            // Update GPU strategy
            const gpuStrategyRadio = document.querySelector(`input[name="gpu-strategy"][value="${config.gpuStrategy}"]`);
            if (gpuStrategyRadio) {
                gpuStrategyRadio.checked = true;
            }
            
            // Update primary model
            const primaryModelSelect = document.getElementById('primary-model');
            if (primaryModelSelect) {
                primaryModelSelect.value = config.primaryModel;
            }
        }
    }
    
    updateGPUStrategyConfiguration(strategy) {
        console.log('Updating GPU strategy configuration:', strategy);
        
        // Update agent GPU assignments based on strategy
        const strategies = {
            balanced: {
                code_generator: 'both',
                research_assistant: 'both',
                document_analyzer: 'both'
            },
            specialized: {
                code_generator: '0',
                research_assistant: 'both',
                document_analyzer: '1'
            },
            high_throughput: {
                code_generator: 'both',
                research_assistant: 'both',
                document_analyzer: 'both'
            },
            single_large: {
                code_generator: 'both',
                research_assistant: 'both',
                document_analyzer: 'both'
            }
        };
        
        const strategyConfig = strategies[strategy];
        if (strategyConfig) {
            Object.entries(strategyConfig).forEach(([agentType, gpu]) => {
                const gpuSelect = document.getElementById(`${agentType}-gpu`);
                if (gpuSelect) {
                    gpuSelect.value = gpu;
                }
            });
        }
    }
    
    async loadCurrentConfiguration() {
        try {
            // Load current system configuration from API
            const response = await this.fetchAPI('/system/configuration');
            if (response) {
                this.applyConfigurationProfile(response);
            }
        } catch (error) {
            console.log('No existing configuration found, using defaults');
            // Load default development configuration
            this.loadConfigurationProfile('development');
        }
    }
    
    async applyConfiguration() {
        const config = this.gatherConfigurationData();
        
        try {
            const response = await this.fetchAPI('/system/configuration', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (response.success) {
                this.showNotification('Configuration applied successfully', 'success');
            } else {
                throw new Error(response.message || 'Failed to apply configuration');
            }
        } catch (error) {
            console.error('Failed to apply configuration:', error);
            this.showNotification('Failed to apply configuration', 'error');
        }
    }
    
    gatherConfigurationData() {
        // Gather all configuration data from the form
        const workMode = document.querySelector('input[name="work-mode"]:checked')?.value;
        const gpuStrategy = document.querySelector('input[name="gpu-strategy"]:checked')?.value;
        const primaryModel = document.getElementById('primary-model')?.value;
        const fallbackModel = document.getElementById('fallback-model')?.value;
        
        // Gather agent configurations
        const agents = {};
        ['code_generator', 'research_assistant', 'document_analyzer'].forEach(agentType => {
            const instancesElement = document.getElementById(`${agentType}-instances`);
            const gpuSelect = document.getElementById(`${agentType}-gpu`);
            const autoscaleCheckbox = document.getElementById(`${agentType}-autoscale`);
            const prioritySelect = document.getElementById(`${agentType}-priority`);
            
            agents[agentType] = {
                instances: parseInt(instancesElement?.textContent || '1'),
                gpu: gpuSelect?.value || 'both',
                autoscale: autoscaleCheckbox?.checked || false,
                priority: parseInt(prioritySelect?.value || '2')
            };
        });
        
        // Gather performance settings
        const performance = {
            reservedMemory: parseFloat(document.getElementById('reserved-memory')?.value || '2'),
            memoryBuffer: parseFloat(document.getElementById('memory-buffer')?.value || '1'),
            maxConcurrentTasks: parseInt(document.getElementById('max-concurrent-tasks')?.value || '8'),
            taskTimeout: parseInt(document.getElementById('task-timeout')?.value || '300'),
            cacheSize: parseFloat(document.getElementById('cache-size')?.value || '4'),
            autoCleanup: document.getElementById('auto-cleanup')?.checked || true,
            batchProcessing: document.getElementById('batch-processing')?.checked || true
        };
        
        // Gather local processing settings
        const localProcessing = {
            fileIndexing: document.getElementById('file-indexing')?.checked || true,
            gitIntegration: document.getElementById('git-integration')?.checked || true,
            shellCommands: document.getElementById('shell-commands')?.checked || true,
            autoScanning: document.getElementById('auto-scanning')?.checked || false,
            allowedDirectories: document.getElementById('allowed-directories')?.value.split('\n').filter(dir => dir.trim()) || []
        };
        
        return {
            profile: {
                name: document.getElementById('current-profile')?.value || 'custom',
                workMode,
                gpuStrategy,
                primaryModel,
                fallbackModel
            },
            agents,
            performance,
            localProcessing,
            timestamp: new Date().toISOString()
        };
    }
    
    resetConfiguration() {
        if (confirm('Are you sure you want to reset to default configuration? This will lose all current settings.')) {
            this.loadConfigurationProfile('development');
            this.showNotification('Configuration reset to defaults', 'info');
        }
    }
    
    previewConfiguration() {
        const config = this.gatherConfigurationData();
        
        // Create preview modal or display
        const preview = JSON.stringify(config, null, 2);
        const previewWindow = window.open('', '_blank', 'width=800,height=600');
        previewWindow.document.write(`
            <html>
                <head><title>Configuration Preview</title></head>
                <body style="font-family: monospace; padding: 20px; background: #1a1a1a; color: #ffffff;">
                    <h2>Configuration Preview</h2>
                    <pre style="background: #2a2a2a; padding: 15px; border-radius: 5px; overflow: auto;">${preview}</pre>
                </body>
            </html>
        `);
    }
    
    saveConfigurationProfile() {
        const profileName = prompt('Enter a name for this configuration profile:');
        if (profileName) {
            const config = this.gatherConfigurationData();
            // Save to local storage or API
            localStorage.setItem(`config_profile_${profileName}`, JSON.stringify(config));
            this.showNotification(`Configuration saved as "${profileName}"`, 'success');
        }
    }
    
    exportConfiguration() {
        const config = this.gatherConfigurationData();
        const dataStr = JSON.stringify(config, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `ai_ecosystem_config_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        this.showNotification('Configuration exported successfully', 'success');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // Research Assistant Methods
    async loadResearchData() {
        console.log('Loading research data...');
        this.initializeResearchInterface();
    }

    initializeResearchInterface() {
        // Initialize workflow filter buttons
        const filterBtns = document.querySelectorAll('.filter-btn');
        filterBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Remove active class from all buttons
                filterBtns.forEach(b => b.classList.remove('active'));
                // Add active class to clicked button
                e.target.classList.add('active');
                
                // Filter workflows
                this.filterWorkflows(e.target.dataset.filter);
            });
        });
        
        // Initialize workflow cards
        const workflowCards = document.querySelectorAll('.workflow-card');
        workflowCards.forEach(card => {
            card.addEventListener('click', (e) => {
                // Remove selected class from all cards
                workflowCards.forEach(c => c.classList.remove('selected'));
                // Add selected class to clicked card
                card.classList.add('selected');
                
                // Update selected workflow display
                const workflowName = card.querySelector('h4').textContent;
                const selectedWorkflowElement = document.getElementById('selected-workflow');
                if (selectedWorkflowElement) {
                    selectedWorkflowElement.textContent = workflowName;
                }
                
                this.updateExecuteButton();
            });
        });
        
        // Initialize backend options
        const backendOptions = document.querySelectorAll('.backend-option');
        backendOptions.forEach(option => {
            option.addEventListener('click', (e) => {
                // Remove selected class from all options
                backendOptions.forEach(o => o.classList.remove('selected'));
                // Add selected class to clicked option
                option.classList.add('selected');
                
                // Update selected backend display
                const backendName = option.querySelector('h4').textContent;
                const selectedBackendElement = document.getElementById('selected-backend');
                if (selectedBackendElement) {
                    selectedBackendElement.textContent = backendName;
                }
                
                this.updateExecuteButton();
            });
        });
        
        // Initialize file tree
        this.initializeFileTree();
        
        // Initialize execution controls
        this.initializeExecutionControls();
        
        // Set default selections
        this.setDefaultSelections();

        // Load workflow types and backend configs
        this.loadWorkflowOptions();
    }

    async loadWorkflowOptions() {
        try {
            const [workflows, backends] = await Promise.all([
                this.fetchAPI('/research-assistant/workflows'),
                this.fetchAPI('/research-assistant/backend-configs')
            ]);

            this.populateWorkflowSelect(workflows);
            this.populateBackendSelect(backends);
        } catch (error) {
            console.error('Failed to load workflow options:', error);
        }
    }

    populateWorkflowSelect(workflows) {
        const select = document.getElementById('workflow-type-select');
        if (select && workflows) {
            select.innerHTML = '<option value="">Select workflow...</option>';
            Object.values(workflows).forEach(workflow => {
                const option = document.createElement('option');
                option.value = workflow.id;
                option.textContent = workflow.name;
                select.appendChild(option);
            });
        }
    }

    populateBackendSelect(backends) {
        const select = document.getElementById('backend-config-select');
        if (select && backends) {
            select.innerHTML = '<option value="">Select backend...</option>';
            Object.values(backends).forEach(backend => {
                const option = document.createElement('option');
                option.value = backend.id;
                option.textContent = backend.name;
                select.appendChild(option);
            });
        }
    }
    
    filterWorkflows(filter) {
        const workflowCards = document.querySelectorAll('.workflow-card');
        
        workflowCards.forEach(card => {
            if (filter === 'all' || card.dataset.category === filter) {
                card.style.display = 'flex';
            } else {
                card.style.display = 'none';
            }
        });
    }
    
    initializeFileTree() {
        const treeNodes = document.querySelectorAll('.tree-node');
        
        treeNodes.forEach(node => {
            node.addEventListener('click', (e) => {
                e.stopPropagation();
                
                const treeItem = node.parentElement;
                const isExpanded = treeItem.classList.contains('expanded');
                
                if (isExpanded) {
                    treeItem.classList.remove('expanded');
                    const icon = node.querySelector('i');
                    if (icon.classList.contains('fa-folder-open')) {
                        icon.classList.remove('fa-folder-open');
                        icon.classList.add('fa-folder');
                    }
                } else {
                    treeItem.classList.add('expanded');
                    const icon = node.querySelector('i');
                    if (icon.classList.contains('fa-folder')) {
                        icon.classList.remove('fa-folder');
                        icon.classList.add('fa-folder-open');
                    }
                }
                
                // Simulate document selection (for demo)
                if (Math.random() > 0.7) { // 30% chance to "select" documents
                    this.addSelectedDocument(node.querySelector('span').textContent);
                }
            });
        });
    }
    
    addSelectedDocument(documentName) {
        const selectedItems = document.getElementById('selected-items');
        const emptySelection = selectedItems.querySelector('.empty-selection');
        
        if (emptySelection) {
            emptySelection.remove();
        }
        
        const documentItem = document.createElement('div');
        documentItem.className = 'selected-document-item';
        documentItem.innerHTML = `
            <div class="document-info">
                <i class="fas fa-file-alt"></i>
                <span>${documentName}</span>
            </div>
            <button class="btn btn-sm btn-secondary" onclick="this.parentElement.remove(); window.aiDashboard.updateDocumentCount();">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        selectedItems.appendChild(documentItem);
        this.updateDocumentCount();
    }
    
    updateDocumentCount() {
        const selectedItems = document.querySelectorAll('.selected-document-item');
        const count = selectedItems.length;
        
        const selectionCountElement = document.querySelector('.selection-count');
        if (selectionCountElement) {
            selectionCountElement.textContent = 
                count === 0 ? 'No documents selected' : `${count} document${count > 1 ? 's' : ''} selected`;
        }
        
        const selectedDocsCountElement = document.getElementById('selected-documents-count');
        if (selectedDocsCountElement) {
            selectedDocsCountElement.textContent = `${count} selected`;
        }
        
        this.updateExecuteButton();
    }
    
    initializeExecutionControls() {
        // Validate Configuration button
        const validateBtn = document.getElementById('validate-config-btn');
        if (validateBtn) {
            validateBtn.addEventListener('click', () => {
                this.validateConfiguration();
            });
        }
        
        // Execute Research button
        const executeBtn = document.getElementById('execute-research-btn');
        if (executeBtn) {
            executeBtn.addEventListener('click', () => {
                this.executeResearchWorkflow();
            });
        }
        
        // Cancel Execution button
        const cancelBtn = document.getElementById('cancel-execution-btn');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => {
                this.cancelExecution();
            });
        }
        
        // Clear Selection button
        const clearBtn = document.getElementById('clear-selection-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearDocumentSelection();
            });
        }
    }
    
    setDefaultSelections() {
        // Select first workflow by default
        const firstWorkflow = document.querySelector('.workflow-card');
        if (firstWorkflow) {
            firstWorkflow.click();
        }
        
        // Select balanced backend by default
        const balancedBackend = document.querySelector('.backend-option[data-backend="balanced"]');
        if (balancedBackend) {
            balancedBackend.click();
        }
    }
    
    updateExecuteButton() {
        const executeBtn = document.getElementById('execute-research-btn');
        const selectedWorkflow = document.querySelector('.workflow-card.selected');
        const selectedBackend = document.querySelector('.backend-option.selected');
        const selectedDocs = document.querySelectorAll('.selected-document-item');
        
        const canExecute = selectedWorkflow && selectedBackend && selectedDocs.length > 0;
        
        if (executeBtn) {
            executeBtn.disabled = !canExecute;
            if (canExecute) {
                executeBtn.classList.remove('btn-secondary');
                executeBtn.classList.add('btn-primary');
            } else {
                executeBtn.classList.remove('btn-primary');
                executeBtn.classList.add('btn-secondary');
            }
        }
    }
    
    validateConfiguration() {
        const selectedWorkflow = document.querySelector('.workflow-card.selected');
        const selectedBackend = document.querySelector('.backend-option.selected');
        const selectedDocs = document.querySelectorAll('.selected-document-item');
        
        let validationMessage = '';
        let isValid = true;
        
        if (!selectedWorkflow) {
            validationMessage += '• Please select a workflow\n';
            isValid = false;
        }
        
        if (!selectedBackend) {
            validationMessage += '• Please select a backend configuration\n';
            isValid = false;
        }
        
        if (selectedDocs.length === 0) {
            validationMessage += '• Please select at least one document\n';
            isValid = false;
        }
        
        if (isValid) {
            this.showNotification('Configuration is valid and ready for execution', 'success');
        } else {
            alert('Configuration Validation Failed:\n\n' + validationMessage);
        }
    }
    
    executeResearchWorkflow() {
        const selectedWorkflow = document.querySelector('.workflow-card.selected');
        const selectedBackend = document.querySelector('.backend-option.selected');
        const selectedDocs = document.querySelectorAll('.selected-document-item');
        
        if (!selectedWorkflow || !selectedBackend || selectedDocs.length === 0) {
            this.validateConfiguration();
            return;
        }
        
        // Show execution status
        const statusPanel = document.getElementById('execution-status');
        if (statusPanel) {
            statusPanel.style.display = 'block';
        }
        
        // Simulate execution progress
        this.simulateExecution();
        
        this.showNotification('Research execution started', 'info');
    }
    
    simulateExecution() {
        const progressBar = document.getElementById('execution-progress');
        const progressText = document.getElementById('execution-progress-text');
        const logContainer = document.getElementById('execution-log');
        
        if (!progressBar || !progressText || !logContainer) return;
        
        let progress = 0;
        const steps = [
            'Initializing workflow...',
            'Loading backend model...',
            'Processing documents...',
            'Analyzing content...',
            'Generating results...',
            'Finalizing output...',
            'Execution completed!'
        ];
        
        const interval = setInterval(() => {
            progress += 15;
            progressBar.style.width = `${Math.min(progress, 100)}%`;
            
            const stepIndex = Math.floor(progress / 15);
            if (stepIndex < steps.length) {
                progressText.textContent = steps[stepIndex];
                
                // Add log entry
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                logEntry.innerHTML = `
                    <span class="log-time">${new Date().toLocaleTimeString().slice(0, 5)}</span>
                    <span class="log-message">${steps[stepIndex]}</span>
                `;
                logContainer.appendChild(logEntry);
                logContainer.scrollTop = logContainer.scrollHeight;
            }
            
            if (progress >= 100) {
                clearInterval(interval);
                progressText.textContent = 'Research completed successfully!';
                this.showNotification('Research execution completed', 'success');
                
                // Hide status after 3 seconds
                setTimeout(() => {
                    const statusPanel = document.getElementById('execution-status');
                    if (statusPanel) {
                        statusPanel.style.display = 'none';
                    }
                }, 3000);
            }
        }, 1000);
    }
    
    cancelExecution() {
        const statusPanel = document.getElementById('execution-status');
        if (statusPanel) {
            statusPanel.style.display = 'none';
        }
        this.showNotification('Research execution cancelled', 'info');
    }
    
    clearDocumentSelection() {
        const selectedItems = document.getElementById('selected-items');
        if (selectedItems) {
            selectedItems.innerHTML = `
                <div class="empty-selection">
                    <i class="fas fa-folder-open"></i>
                    <p>No documents selected</p>
                </div>
            `;
        }
        this.updateDocumentCount();
    }
    
    showWorkflowInterface(category) {
        const categorySelection = document.getElementById('category-selection');
        const workflowInterface = document.getElementById('workflow-interface');
        
        if (categorySelection && workflowInterface) {
            categorySelection.style.display = 'none';
            workflowInterface.style.display = 'block';
            
            // Update title based on category
            const title = document.getElementById('current-workflow-title');
            if (title) {
                const categoryNames = {
                    'document_processing': 'Document Processing',
                    'content_generation': 'Content Generation',
                    'technical_analysis': 'Technical Analysis'
                };
                title.textContent = categoryNames[category] || 'Workflow';
            }
        }
    }

    showCategorySelection() {
        const categorySelection = document.getElementById('category-selection');
        const workflowInterface = document.getElementById('workflow-interface');
        
        if (categorySelection && workflowInterface) {
            categorySelection.style.display = 'block';
            workflowInterface.style.display = 'none';
        }
    }

    async executeWorkflow() {
        const workflowType = document.getElementById('workflow-type-select').value;
        const backendConfig = document.getElementById('backend-config-select').value;
        const saveResults = document.getElementById('save-results').checked;
        const outputFormat = document.getElementById('output-format').value;

        if (!workflowType || !backendConfig) {
            alert('Please select both workflow type and backend configuration');
            return;
        }

        const resultsContainer = document.getElementById('workflow-results');
        resultsContainer.innerHTML = '<div class="loading-state"><i class="fas fa-spinner"></i><p>Executing workflow...</p></div>';

        try {
            const response = await this.fetchAPI('/research-assistant/execute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    workflow_type: workflowType,
                    backend_config: backendConfig,
                    save_results: saveResults,
                    output_format: outputFormat,
                    input_documents: [], // TODO: Get from file input
                    input_folders: []
                })
            });

            this.displayWorkflowResults(response);
        } catch (error) {
            console.error('Workflow execution failed:', error);
            resultsContainer.innerHTML = '<div class="error-state"><i class="fas fa-exclamation-triangle"></i><p>Workflow execution failed</p></div>';
        }
    }

    displayWorkflowResults(results) {
        const container = document.getElementById('workflow-results');
        if (results && results.results) {
            container.innerHTML = `
                <div class="results-content">
                    <h4>Workflow Results</h4>
                    <div class="result-summary">
                        <pre>${JSON.stringify(results.results, null, 2)}</pre>
                    </div>
                    <div class="result-metadata">
                        <p><strong>Status:</strong> ${results.status}</p>
                        <p><strong>Created:</strong> ${new Date(results.created_at).toLocaleString()}</p>
                        ${results.completed_at ? `<p><strong>Completed:</strong> ${new Date(results.completed_at).toLocaleString()}</p>` : ''}
                    </div>
                </div>
            `;
        }
    }

    // LLM Backends Methods
    async loadBackendsData() {
        console.log('Loading backends data...');
        this.initializeBackendsInterface();
        await this.loadRegisteredBackends();
    }

    initializeBackendsInterface() {
        // Initialize discover backends button
        const discoverBtn = document.getElementById('discover-backends-btn');
        if (discoverBtn) {
            discoverBtn.addEventListener('click', () => {
                this.discoverBackends();
            });
        }

        // Initialize add backend button
        const addBackendBtn = document.getElementById('add-backend-btn');
        if (addBackendBtn) {
            addBackendBtn.addEventListener('click', () => {
                this.showAddBackendDialog();
            });
        }

        // Initialize backend selector for model management
        const backendSelector = document.getElementById('backend-selector');
        if (backendSelector) {
            backendSelector.addEventListener('change', (e) => {
                this.loadBackendModels(e.target.value);
            });
        }

        // Initialize test interface
        const testBtn = document.getElementById('test-generate-btn');
        if (testBtn) {
            testBtn.addEventListener('click', () => {
                this.testGeneration();
            });
        }

        // Update test backend options
        this.updateTestBackendOptions();
    }

    async discoverBackends() {
        const resultsContainer = document.getElementById('discovery-results');
        resultsContainer.innerHTML = '<div class="loading-state"><i class="fas fa-spinner"></i><p>Scanning for backends...</p></div>';

        try {
            const discovered = await this.fetchAPI('/llm-backends/discover');
            this.displayDiscoveredBackends(discovered);
        } catch (error) {
            console.error('Backend discovery failed:', error);
            resultsContainer.innerHTML = '<div class="error-state"><i class="fas fa-exclamation-triangle"></i><p>Discovery failed</p></div>';
        }
    }

    displayDiscoveredBackends(backends) {
        const container = document.getElementById('discovery-results');
        if (backends && backends.length > 0) {
            container.innerHTML = backends.map(backend => `
                <div class="discovered-backend">
                    <div class="discovered-info">
                        <h4>${backend.name}</h4>
                        <p>${backend.endpoint_url} - ${backend.backend_type}</p>
                    </div>
                    <button class="btn btn-primary btn-sm" onclick="aiDashboard.createBackendFromDiscovered('${backend.name}', '${backend.backend_type}', '${backend.endpoint_url}')">
                        Add Backend
                    </button>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-search"></i><p>No backends discovered</p></div>';
        }
    }

    async createBackendFromDiscovered(name, type, url) {
        try {
            await this.fetchAPI('/llm-backends/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    instance_name: name,
                    backend_type: type,
                    config: {
                        base_url: url
                    }
                })
            });

            await this.loadRegisteredBackends();
            this.updateTestBackendOptions();
        } catch (error) {
            console.error('Failed to create backend:', error);
            alert('Failed to create backend');
        }
    }

    async loadRegisteredBackends() {
        try {
            const backends = await this.fetchAPI('/llm-backends/list');
            this.displayRegisteredBackends(backends);
            this.updateBackendSelector(backends);
        } catch (error) {
            console.error('Failed to load backends:', error);
        }
    }

    displayRegisteredBackends(backends) {
        const container = document.getElementById('backends-grid');
        if (backends && Object.keys(backends.backends).length > 0) {
            container.innerHTML = Object.entries(backends.backends).map(([name, backend]) => `
                <div class="backend-card">
                    <div class="backend-status-indicator ${backend.status}"></div>
                    <div class="backend-header">
                        <div class="backend-type">${backend.backend_type}</div>
                        <div class="backend-status">
                            <i class="fas fa-circle"></i>
                            ${backend.status}
                        </div>
                    </div>
                    <div class="backend-info">
                        <h4>${name}</h4>
                        <p>${backend.description}</p>
                    </div>
                    <div class="backend-capabilities">
                        ${backend.capabilities.map(cap => `<span class="capability-tag">${cap}</span>`).join('')}
                    </div>
                    <div class="backend-actions">
                        <button class="btn btn-sm btn-primary" onclick="aiDashboard.setActiveBackend('${name}')">
                            Set Active
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="aiDashboard.removeBackend('${name}')">
                            Remove
                        </button>
                    </div>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-server"></i><p>No backends registered yet</p></div>';
        }
    }

    updateBackendSelector(backends) {
        const selector = document.getElementById('backend-selector');
        if (selector && backends) {
            selector.innerHTML = '<option value="">Select a backend...</option>';
            Object.keys(backends.backends).forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                selector.appendChild(option);
            });
        }
    }

    async setActiveBackend(name) {
        try {
            await this.fetchAPI(`/llm-backends/set-active/${name}`, {
                method: 'POST'
            });
            await this.loadRegisteredBackends();
        } catch (error) {
            console.error('Failed to set active backend:', error);
            alert('Failed to set active backend');
        }
    }

    async removeBackend(name) {
        if (confirm(`Are you sure you want to remove backend "${name}"?`)) {
            try {
                await this.fetchAPI(`/llm-backends/${name}`, {
                    method: 'DELETE'
                });
                await this.loadRegisteredBackends();
                this.updateTestBackendOptions();
            } catch (error) {
                console.error('Failed to remove backend:', error);
                alert('Failed to remove backend');
            }
        }
    }

    async loadBackendModels(backendName) {
        if (!backendName) {
            document.getElementById('models-by-backend').innerHTML = '<div class="empty-state"><i class="fas fa-brain"></i><p>Select a backend to manage models</p></div>';
            return;
        }

        try {
            const models = await this.fetchAPI(`/llm-backends/models?backend=${backendName}`);
            this.displayBackendModels(backendName, models);
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    displayBackendModels(backendName, models) {
        const container = document.getElementById('models-by-backend');
        if (models && models.models && models.models.length > 0) {
            container.innerHTML = `
                <div class="backend-models">
                    <h4><i class="fas fa-server"></i> ${backendName} Models</h4>
                    <div class="models-list">
                        ${models.models.map(model => `
                            <div class="model-item">
                                <div class="model-name">${model.name}</div>
                                <div class="model-actions">
                                    <button class="btn btn-sm btn-primary" onclick="aiDashboard.loadModel('${backendName}', '${model.name}')">
                                        Load
                                    </button>
                                    <button class="btn btn-sm btn-secondary" onclick="aiDashboard.unloadModel('${backendName}', '${model.name}')">
                                        Unload
                                    </button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        } else {
            container.innerHTML = '<div class="empty-state"><i class="fas fa-brain"></i><p>No models available</p></div>';
        }
    }

    async loadModel(backend, model) {
        try {
            await this.fetchAPI('/llm-backends/models/load', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    backend: backend,
                    model: model
                })
            });
            
            this.updateTestModelOptions();
        } catch (error) {
            console.error('Failed to load model:', error);
            alert('Failed to load model');
        }
    }

    async unloadModel(backend, model) {
        try {
            await this.fetchAPI('/llm-backends/models/unload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    backend: backend,
                    model: model
                })
            });
        } catch (error) {
            console.error('Failed to unload model:', error);
            alert('Failed to unload model');
        }
    }

    updateTestBackendOptions() {
        // This would populate the test interface backend selector
        // Implementation would be similar to updateBackendSelector
    }

    updateTestModelOptions() {
        // This would populate the test interface model selector
        // Based on loaded models in the selected backend
    }

    async testGeneration() {
        const backend = document.getElementById('test-backend-select').value;
        const model = document.getElementById('test-model-select').value;
        const prompt = document.getElementById('test-prompt').value;

        if (!backend || !model || !prompt) {
            alert('Please select backend, model, and enter a prompt');
            return;
        }

        const resultsContainer = document.getElementById('test-results');
        resultsContainer.innerHTML = '<div class="loading-state"><i class="fas fa-spinner"></i><p>Generating response...</p></div>';

        try {
            const response = await this.fetchAPI('/llm-backends/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    model: model,
                    backend: backend
                })
            });

            this.displayTestResults(response);
        } catch (error) {
            console.error('Generation failed:', error);
            resultsContainer.innerHTML = '<div class="error-state"><i class="fas fa-exclamation-triangle"></i><p>Generation failed</p></div>';
        }
    }

    displayTestResults(response) {
        const container = document.getElementById('test-results');
        container.innerHTML = `
            <div class="response-content">${response.text}</div>
            <div class="response-metadata">
                <div class="metadata-item">
                    <label>Backend</label>
                    <span>${response.backend}</span>
                </div>
                <div class="metadata-item">
                    <label>Model</label>
                    <span>${response.model}</span>
                </div>
                <div class="metadata-item">
                    <label>Tokens</label>
                    <span>${response.tokens_used}</span>
                </div>
                <div class="metadata-item">
                    <label>Time</label>
                    <span>${response.processing_time}s</span>
                </div>
            </div>
        `;
    }

    showAddBackendDialog() {
        // This would show a modal dialog for manually adding backends
        // For now, just show an alert
        alert('Manual backend addition coming soon. Use the discovery feature for now.');
    }

    // System metrics update handler
    updateSystemMetrics(metrics) {
        console.log('Updating system metrics:', metrics);
        // Handle system metrics updates - placeholder for now
    }

    // Cleanup method
    destroy() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }

    // ===== RESEARCH CATEGORIES FUNCTIONALITY =====
    
    setupResearchCategories() {
        // Category selection handlers
        const categoryContainers = document.querySelectorAll('.category-container');
        console.log(`Found ${categoryContainers.length} category containers`);
        
        if (categoryContainers.length === 0) {
            console.error('No category containers found! Checking HTML structure...');
            const researchSection = document.getElementById('research');
            console.log('Research section:', researchSection);
            if (researchSection) {
                console.log('Research section HTML:', researchSection.innerHTML.substring(0, 500));
            }
            return;
        }
        
        categoryContainers.forEach((container, index) => {
            console.log(`Setting up category container ${index}:`, container);
            
            container.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const category = container.getAttribute('data-category');
                console.log(`Clicked category: ${category}`);
                this.showWorkspace(category);
            });
            
            // Add visual feedback
            container.style.cursor = 'pointer';
            container.style.userSelect = 'none';
        });

        // Back to categories handlers
        const backButtons = document.querySelectorAll('[id^="back-to-categories"]');
        console.log(`Found ${backButtons.length} back buttons`);
        backButtons.forEach((button, index) => {
            console.log(`Setting up back button ${index}:`, button.id);
            button.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('Back to categories clicked from:', button.id);
                this.showCategories();
            });
        });

        // Text Generation Workspace
        this.setupTextGenerationWorkspace();
        
        // Technical Analysis Workspace
        this.setupTechnicalAnalysisWorkspace();
        
        // Document Processing Workspace
        this.setupDocumentProcessingWorkspace();
    }

    showWorkspace(category) {
        console.log(`Showing workspace for category: ${category}`);
        
        // Hide categories
        const categoriesSection = document.getElementById('research-categories');
        console.log('Categories section:', categoriesSection);
        if (categoriesSection) {
            categoriesSection.style.display = 'none';
            console.log('Categories section hidden');
        } else {
            console.error('Categories section not found!');
        }

        // Show appropriate workspace
        const workspaceId = `${category.replace('_', '-')}-workspace`;
        console.log(`Looking for workspace: ${workspaceId}`);
        const workspace = document.getElementById(workspaceId);
        console.log('Workspace element:', workspace);
        if (workspace) {
            workspace.style.display = 'block';
            console.log(`Workspace ${workspaceId} shown`);
        } else {
            console.error(`Workspace ${workspaceId} not found!`);
        }
    }

    showCategories() {
        console.log('Showing categories...');
        
        // Show categories
        const categoriesSection = document.getElementById('research-categories');
        console.log('Categories section:', categoriesSection);
        if (categoriesSection) {
            categoriesSection.style.display = 'block';
            console.log('Categories section shown');
        } else {
            console.error('Categories section not found!');
        }

        // Hide all workspaces
        const workspaces = document.querySelectorAll('.workspace-interface');
        console.log(`Found ${workspaces.length} workspaces to hide`);
        workspaces.forEach(workspace => {
            workspace.style.display = 'none';
            console.log(`Hidden workspace: ${workspace.id}`);
        });
    }

    setupTextGenerationWorkspace() {
        const sendButton = document.getElementById('send-text-gen');
        const inputTextarea = document.getElementById('text-gen-input');
        const messagesContainer = document.getElementById('text-gen-messages');
        const fileInput = document.getElementById('text-gen-files');
        const uploadArea = document.getElementById('text-gen-upload');

        // Send message handler
        if (sendButton && inputTextarea) {
            const sendMessage = async () => {
                const message = inputTextarea.value.trim();
                if (!message) return;

                // Add user message to chat
                this.addChatMessage(messagesContainer, message, 'user');
                inputTextarea.value = '';

                // Get generation settings
                const contentType = document.getElementById('content-type')?.value || 'summary';
                const tone = document.getElementById('content-tone')?.value || 'professional';
                const length = document.getElementById('content-length')?.value || 'medium';

                try {
                    // Call API for text generation
                    const response = await this.generateText(message, {
                        type: contentType,
                        tone: tone,
                        length: length
                    });

                    // Add assistant response
                    this.addChatMessage(messagesContainer, response, 'assistant');
                } catch (error) {
                    this.addChatMessage(messagesContainer, 'Sorry, I encountered an error processing your request.', 'assistant');
                }
            };

            sendButton.addEventListener('click', sendMessage);
            inputTextarea.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        }

        // File upload handler
        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('drag-over');
            });
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                const files = Array.from(e.dataTransfer.files);
                this.handleFileUpload(files, 'text-gen');
            });

            fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.handleFileUpload(files, 'text-gen');
            });
        }
    }

    setupTechnicalAnalysisWorkspace() {
        const startAnalysisButton = document.getElementById('start-analysis');
        
        if (startAnalysisButton) {
            startAnalysisButton.addEventListener('click', async () => {
                const analysisType = document.getElementById('analysis-type')?.value || 'comparative';
                const analysisDepth = document.getElementById('analysis-depth')?.value || 'standard';
                
                try {
                    await this.startTechnicalAnalysis(analysisType, analysisDepth);
                } catch (error) {
                    console.error('Technical analysis failed:', error);
                }
            });
        }
    }

    setupDocumentProcessingWorkspace() {
        const batchFileInput = document.getElementById('batch-file-input');
        const uploadZone = document.getElementById('batch-upload-zone');
        const browseButton = document.getElementById('browse-batch-files');
        const startProcessingButton = document.getElementById('start-batch-processing');

        // File upload handlers
        if (browseButton && batchFileInput) {
            browseButton.addEventListener('click', () => batchFileInput.click());
        }

        if (uploadZone) {
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('drag-over');
            });
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('drag-over');
            });
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('drag-over');
                const files = Array.from(e.dataTransfer.files);
                this.handleBatchUpload(files);
            });
        }

        if (batchFileInput) {
            batchFileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.handleBatchUpload(files);
            });
        }

        // Processing handler
        if (startProcessingButton) {
            startProcessingButton.addEventListener('click', () => {
                this.startBatchProcessing();
            });
        }
    }

    addChatMessage(container, message, type) {
        if (!container) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = `<p>${message}</p>`;
        
        messageDiv.appendChild(messageContent);
        container.appendChild(messageDiv);
        
        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    async generateText(prompt, options) {
        const response = await fetch(`${this.apiBaseUrl}/generate-text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                options: options
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.generated_text || 'No response generated.';
    }

    async startTechnicalAnalysis(type, depth) {
        // Update UI to show analysis in progress
        const chartsArea = document.getElementById('analysis-charts');
        if (chartsArea) {
            chartsArea.innerHTML = `
                <div class="analysis-progress">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Running ${type} analysis...</p>
                </div>
            `;
        }

        const response = await fetch(`${this.apiBaseUrl}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                type: type,
                depth: depth
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        this.displayAnalysisResults(data);
    }

    displayAnalysisResults(data) {
        const chartsArea = document.getElementById('analysis-charts');
        if (chartsArea) {
            chartsArea.innerHTML = `
                <div class="analysis-results">
                    <h5>Analysis Complete</h5>
                    <p>Results: ${data.summary || 'Analysis completed successfully.'}</p>
                </div>
            `;
        }

        // Update overview cards
        this.updateOverviewCards(data);
    }

    updateOverviewCards(data) {
        const docsAnalyzed = document.getElementById('docs-analyzed');
        const dataPoints = document.getElementById('data-points');
        const insights = document.getElementById('insights-generated');
        const confidence = document.getElementById('confidence-score');

        if (docsAnalyzed) docsAnalyzed.textContent = data.documents_analyzed || '0';
        if (dataPoints) dataPoints.textContent = data.data_points || '0';
        if (insights) insights.textContent = data.insights_count || '0';
        if (confidence) confidence.textContent = `${data.confidence_score || 0}%`;
    }

    handleFileUpload(files, context) {
        const fileListId = `${context}-file-list`;
        const fileList = document.getElementById(fileListId);
        
        if (fileList) {
            files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'uploaded-file-item';
                fileItem.innerHTML = `
                    <i class="fas fa-file"></i>
                    <span>${file.name}</span>
                    <button class="btn btn-sm btn-secondary" onclick="this.parentElement.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                `;
                fileList.appendChild(fileItem);
            });
        }
    }

    handleBatchUpload(files) {
        const queueList = document.getElementById('processing-queue-list');
        
        if (queueList) {
            // Clear empty state
            queueList.innerHTML = '';
            
            files.forEach(file => {
                const queueItem = document.createElement('div');
                queueItem.className = 'queue-item';
                queueItem.innerHTML = `
                    <div class="queue-item-info">
                        <i class="fas fa-file"></i>
                        <span>${file.name}</span>
                    </div>
                    <div class="queue-item-status">
                        <span class="status pending">Pending</span>
                    </div>
                `;
                queueList.appendChild(queueItem);
            });
        }
    }

    async startBatchProcessing() {
        const workflow = document.getElementById('processing-workflow')?.value || 'extract_text';
        const outputFormat = document.getElementById('processing-output')?.value || 'json';
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/process-batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    workflow: workflow,
                    output_format: outputFormat
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.updateProcessingStatus(data);
        } catch (error) {
            console.error('Batch processing failed:', error);
        }
    }

    updateProcessingStatus(data) {
        // Update queue items status
        const queueItems = document.querySelectorAll('.queue-item');
        queueItems.forEach((item, index) => {
            const status = item.querySelector('.status');
            if (status) {
                status.textContent = 'Processing';
                status.className = 'status processing';
            }
        });
    }
}

// Global functions for agent configuration controls
function adjustInstances(agentType, delta) {
    const instanceElement = document.getElementById(`${agentType}-instances`);
    if (instanceElement) {
        let current = parseInt(instanceElement.textContent);
        let newValue = Math.max(0, Math.min(10, current + delta));
        instanceElement.textContent = newValue;
    }
}

function adjustValue(elementId, delta) {
    const element = document.getElementById(elementId);
    if (element) {
        let current = parseInt(element.value);
        let min = parseInt(element.min) || 0;
        let max = parseInt(element.max) || 100;
        let newValue = Math.max(min, Math.min(max, current + delta));
        element.value = newValue;
    }
}

function configureAgent(agentType) {
    // Open agent-specific configuration modal
    const agentConfigs = {
        code_generator: {
            title: 'Code Generator Configuration',
            options: ['Memory Limit', 'CPU Cores', 'Context Length', 'Model Specialization']
        },
        research_assistant: {
            title: 'Research Assistant Configuration', 
            options: ['Document Types', 'Analysis Depth', 'Output Formats', 'Cache Settings']
        },
        document_analyzer: {
            title: 'Document Analyzer Configuration',
            options: ['Supported Formats', 'Processing Speed', 'Extraction Methods', 'Quality Settings']
        }
    };
    
    const config = agentConfigs[agentType];
    if (config) {
        alert(`${config.title}\n\nAvailable options:\n${config.options.join('\n')}\n\nAdvanced configuration coming soon...`);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiDashboard = new AIEcosystemDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.aiDashboard) {
        window.aiDashboard.destroy();
    }
});