// Fresh start - JavaScript will be added step by step 

// Socket.IO connection
let socket;
let charts = {};

// Global timer variables
let timerInterval = null;
let timerRunning = false;
let elapsedTime = 0; // in seconds

// Global timer functions
function startTimer() {
    if (timerInterval) return; // Already running
    
    timerRunning = true;
    timerInterval = setInterval(updateTimer, 1000);
    console.log('Timer started');
}

function pauseTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        timerRunning = false;
        console.log('Timer paused at:', elapsedTime, 'seconds');
    }
}

function resetTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
    timerRunning = false;
    elapsedTime = 0;
    updateTimeDisplay(0);
    console.log('Timer reset to 00:00:00');
}

function updateTimer() {
    if (timerRunning) {
        elapsedTime += 1;
        updateTimeDisplay(elapsedTime);
    }
}

function setTimerFromServer(serverElapsed, serverRunning) {
    console.log(`🔄 setTimerFromServer called with: elapsed=${serverElapsed}, running=${serverRunning}`);
    
    elapsedTime = Math.floor(serverElapsed) || 0;
    timerRunning = serverRunning;
    
    console.log(`⏱️ Timer sync from server: ${elapsedTime}s, running: ${timerRunning}`);
    
    // Start or stop timer based on server state
    if (serverRunning && !timerInterval) {
        timerInterval = setInterval(updateTimer, 1000);
        console.log('⏱️ Timer started from server sync');
    } else if (!serverRunning && timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        console.log('⏱️ Timer stopped from server sync');
    }
    
    updateTimeDisplay(elapsedTime);
}

function updateTimeDisplay(totalSeconds) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    
    const timeString = 
        String(hours).padStart(2, '0') + ':' +
        String(minutes).padStart(2, '0') + ':' +
        String(seconds).padStart(2, '0');
    
    const display = document.querySelector('.time-display');
    if (display) {
        display.textContent = timeString;
        console.log('⏰ Timer display updated:', timeString);
    } else {
        console.log('❌ Timer display element not found');
    }
}

// Progress bar update function (for backend integration)
function updateProgress(percentage) {
    const progressFill = document.querySelector('.progress-fill');
    const progressPercentage = document.querySelector('.progress-percentage');
    
    if (progressFill && progressPercentage) {
        // Ensure percentage is between 0 and 100
        percentage = Math.max(0, Math.min(100, percentage));
        
        // Round to nearest integer for display (10.1% = 10%, 10.6% = 11%)
        const displayPercentage = Math.round(percentage);
        
        // Update progress bar width with exact percentage (1% accuracy)
        progressFill.style.width = percentage + '%';
        progressFill.setAttribute('data-progress', percentage);
        
        // Update percentage text with rounded integer
        progressPercentage.textContent = displayPercentage + '%';
        
        console.log('Progress updated to:', percentage + '% (displayed as ' + displayPercentage + '%)');
    }
}

// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing tabs...');
    
    // Initialize Socket.IO connection
    initializeSocketIO();
    
    // Initialize charts
    initializeCharts();
    
    const tabs = document.querySelectorAll('.tab-item');
    const arrow = document.querySelector('.tab-arrow-indicator');
    const tabContents = document.querySelectorAll('.tab-content');
    
    console.log('Found tabs:', tabs.length);
    console.log('Found arrow:', arrow ? 'yes' : 'no');
    console.log('Found tab contents:', tabContents.length);
    
    // Position calculation constants
    const logoHeight = 30;             // Approximate logo height
    const tabNavigationMargin = 200;   // Tab navigation margin-top
    const tabSpacing = 100;             // Distance between each tab center
    const arrowOffset = 25;             // Additional offset for fine-tuning (adjust this!)
    
    // Calculate base position
    const basePosition = logoHeight + tabNavigationMargin;
    
    function moveArrowToTab(tabIndex) {
        if (!arrow) return;
        
        // Calculate position for the tab
        const targetPosition = basePosition + (tabIndex * tabSpacing) + arrowOffset;
        
        console.log(`Moving arrow to tab ${tabIndex}, position: ${targetPosition}px`);
        arrow.style.top = targetPosition + 'px';
    }
    
    function switchToTab(targetTab) {
        console.log('Switching to tab:', targetTab);
        
        // Remove active class from all tabs
        tabs.forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Hide all tab contents
        tabContents.forEach(content => {
            content.classList.remove('active');
        });
        
        // Add active class to clicked tab
        const activeTab = document.querySelector(`[data-tab="${targetTab}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
            
            // Show corresponding tab content
            const activeContent = document.getElementById(`${targetTab}-content`);
            if (activeContent) {
                activeContent.classList.add('active');
            }
            
            // Move arrow to active tab
            const tabIndex = Array.from(tabs).indexOf(activeTab);
            moveArrowToTab(tabIndex);
            
            // Update architecture visualization if switching to attack tab
            if (targetTab === 'attack') {
                setTimeout(() => {
                    if (typeof updateComponentPositions !== 'undefined') {
                        updateComponentPositions();
                    }
                }, 100);
            }
        }
    }
    
    // Add click event listeners to tabs
    tabs.forEach((tab, index) => {
        tab.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            console.log(`Tab clicked: ${tabName} (index: ${index})`);
            switchToTab(tabName);
        });
    });
    
    // Initialize arrow position for the active tab
    const activeTab = document.querySelector('.tab-item.active');
    if (activeTab) {
        const activeIndex = Array.from(tabs).indexOf(activeTab);
        moveArrowToTab(activeIndex);
    }
    
    // Percentage button functionality
    const percentageButtons = document.querySelectorAll('.percentage-btn');
    
    percentageButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove selected class from all buttons
            percentageButtons.forEach(btn => {
                btn.classList.remove('selected');
            });
            
            // Add selected class to clicked button
            this.classList.add('selected');
            
            const percentage = parseInt(this.getAttribute('data-percentage'));
            console.log('Selected percentage:', percentage + '%');
            
            // Send configuration update to server
            if (socket) {
                socket.emit('update_config', {
                    type: 'data_percentage',
                    value: percentage
                });
            }
        });
    });
    
    // Control button functionality
    const controlButtons = document.querySelectorAll('.control-btn');
    
    controlButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove selected class from all buttons
            controlButtons.forEach(btn => {
                btn.classList.remove('selected');
            });
            
            // Add selected class to clicked button
            this.classList.add('selected');
            
            const action = this.getAttribute('data-action');
            console.log('Control action:', action);
            
            // Send training control command to server
            if (socket) {
                socket.emit('training_control', {
                    action: action
                });
            }
            
            // Timer will be controlled by server responses
            console.log(`${action} requested - server will handle timer and process management`);
        });
    });
    
    // Timer functions are now global (defined above)
    
    // Helper functions for setting time (kept for compatibility)
    function setRunningTime(totalSeconds) {
        elapsedTime = totalSeconds;
        updateTimeDisplay(totalSeconds);
        console.log('Running time set to:', totalSeconds, 'seconds');
    }
    
    function setRunningTimeFromString(timeString) {
        const parts = timeString.split(':');
        if (parts.length === 3) {
            const hours = parseInt(parts[0]) || 0;
            const minutes = parseInt(parts[1]) || 0;
            const seconds = parseInt(parts[2]) || 0;
            const totalSeconds = hours * 3600 + minutes * 60 + seconds;
            setRunningTime(totalSeconds);
        }
    }

    
    // Example: Simulate progress updates (remove this in production)
    // Uncomment the following lines to test progress bar animation
    /*
    let currentProgress = 0;
    setInterval(() => {
        currentProgress += 0.5; // Test with decimal increments
        if (currentProgress > 100) currentProgress = 0;
        updateProgress(currentProgress);
    }, 100);
    */
    
    // Backend Integration Examples:
    // updateProgress(10.1);  // Shows as 10%
    // updateProgress(10.6);  // Shows as 11%
    // updateProgress(45.3);  // Shows as 45%
    // updateProgress(67.8);  // Shows as 68%
    
    // Slider functionality
    function initializeSliders() {
        // Rounds slider
        const roundsSlider = document.getElementById('rounds-slider');
        const roundsInput = document.getElementById('rounds-input');
        const roundsTrack = document.querySelector('.rounds-track');
        const roundsThumb = document.querySelector('.rounds-thumb');
        
        // Epochs slider
        const epochsSlider = document.getElementById('epochs-slider');
        const epochsInput = document.getElementById('epochs-input');
        const epochsTrack = document.querySelector('.epochs-track');
        const epochsThumb = document.querySelector('.epochs-thumb');
        
        // Update slider progress visual
        function updateSliderProgress(slider, track, thumb, value) {
            const min = parseInt(slider.min);
            const max = parseInt(slider.max);
            const percentage = ((value - min) / (max - min)) * 100;
            
            // Update CSS custom properties for both track and thumb
            track.style.setProperty('--progress', percentage + '%');
            thumb.style.setProperty('--progress', percentage + '%');
            
            // Force a repaint to ensure smooth transition
            track.offsetHeight;
        }
        
        // Rounds slider events
        if (roundsSlider && roundsInput) {
            // Initialize
            updateSliderProgress(roundsSlider, roundsTrack, roundsThumb, roundsSlider.value);
            
            roundsSlider.addEventListener('input', function() {
                roundsInput.value = this.value;
                updateSliderProgress(this, roundsTrack, roundsThumb, this.value);
                console.log('Rounds set to:', this.value);
                
                // Send configuration update to server
                if (socket) {
                    socket.emit('update_config', {
                        type: 'federated_rounds',
                        value: parseInt(this.value)
                    });
                }
            });
            
            roundsInput.addEventListener('input', function() {
                const value = Math.max(1, Math.min(20, parseInt(this.value) || 1));
                this.value = value;
                roundsSlider.value = value;
                updateSliderProgress(roundsSlider, roundsTrack, roundsThumb, value);
                console.log('Rounds set to:', value);
                
                // Send configuration update to server
                if (socket) {
                    socket.emit('update_config', {
                        type: 'federated_rounds',
                        value: value
                    });
                }
            });
        }
        
        // Epochs slider events
        if (epochsSlider && epochsInput) {
            // Initialize
            updateSliderProgress(epochsSlider, epochsTrack, epochsThumb, epochsSlider.value);
            
            epochsSlider.addEventListener('input', function() {
                epochsInput.value = this.value;
                updateSliderProgress(this, epochsTrack, epochsThumb, this.value);
                console.log('Epochs set to:', this.value);
                
                // Send configuration update to server
                if (socket) {
                    socket.emit('update_config', {
                        type: 'epochs_per_round',
                        value: parseInt(this.value)
                    });
                }
            });
            
            epochsInput.addEventListener('input', function() {
                const value = Math.max(1, Math.min(20, parseInt(this.value) || 1));
                this.value = value;
                epochsSlider.value = value;
                updateSliderProgress(epochsSlider, epochsTrack, epochsThumb, value);
                console.log('Epochs set to:', value);
                
                // Send configuration update to server
                if (socket) {
                    socket.emit('update_config', {
                        type: 'epochs_per_round',
                        value: value
                    });
                }
            });
        }
    }
    
    // Initialize sliders
    initializeSliders();
    
    // Dataset selection functionality
    const datasets = document.querySelectorAll('.dataset-box');
    datasets.forEach(dataset => {
        dataset.addEventListener('click', function() {
            // Remove active from all datasets
            datasets.forEach(d => d.classList.remove('active'));
            // Add active to clicked dataset
            this.classList.add('active');
            console.log('Dataset selected:', this.querySelector('.dataset-name').textContent);
            // Calculate sample counts based on new selection
            calculateSampleCounts();
        });
    });
    
    // Update percentage button functionality to trigger sample calculation
    const dataPercentageButtons = document.querySelectorAll('.data-percentage-btn');
    dataPercentageButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove selected class from all buttons
            dataPercentageButtons.forEach(btn => {
                btn.classList.remove('selected');
            });
            
            // Add selected class to clicked button
            this.classList.add('selected');
            
            console.log('Selected data percentage:', this.textContent);
            // Calculate sample counts based on new percentage
            calculateSampleCounts();
        });
    });
    
    // Initialize sample counts on page load
    calculateSampleCounts();
    
    // Initialize architecture visualization
    setTimeout(initializeArchitectureVisualization, 500);
});

// Function to update sample counts from backend
function updateSampleCounts(trainingSamples, validationSamples, testSamples) {
    document.getElementById('training-samples').textContent = trainingSamples || '-';
    document.getElementById('validation-samples').textContent = validationSamples || '-';
    document.getElementById('test-samples').textContent = testSamples || '-';
    console.log('Sample counts updated:', { trainingSamples, validationSamples, testSamples });
}

// Function to reset sample counts to '-'
function resetSampleCounts() {
    document.getElementById('training-samples').textContent = '-';
    document.getElementById('validation-samples').textContent = '-';
    document.getElementById('test-samples').textContent = '-';
    console.log('Sample counts reset to default');
}

// Function to calculate and display sample counts based on dataset and percentage
function calculateSampleCounts() {
    const selectedDataset = document.querySelector('.dataset-box.active');
    const selectedPercentage = document.querySelector('.data-percentage-btn.selected');
    
    if (!selectedDataset || !selectedPercentage) {
        resetSampleCounts();
        return;
    }
    
    const datasetName = selectedDataset.querySelector('.dataset-name').textContent;
    const percentage = parseInt(selectedPercentage.textContent.replace('%', '')) / 100;
    
    // Dataset sample counts (you can adjust these based on actual dataset sizes)
    const datasetSizes = {
        'HAM-10K': { total: 10015, train: 0.7, val: 0.15, test: 0.15 },
        'MNIST': { total: 70000, train: 0.7, val: 0.15, test: 0.15 },
        'CIFAR-10': { total: 60000, train: 0.7, val: 0.15, test: 0.15 },
        'COCO-2017': { total: 123287, train: 0.7, val: 0.15, test: 0.15 }
    };
    
    const dataset = datasetSizes[datasetName];
    if (dataset) {
        const totalSamples = Math.floor(dataset.total * percentage);
        const trainingSamples = Math.floor(totalSamples * dataset.train);
        const validationSamples = Math.floor(totalSamples * dataset.val);
        const testSamples = Math.floor(totalSamples * dataset.test);
        
        updateSampleCounts(trainingSamples, validationSamples, testSamples);
    }
}

// Protection Controls Functionality
document.addEventListener('DOMContentLoaded', function() {
    const protectionButtons = document.querySelectorAll('.protection-btn');
    
    protectionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const action = this.getAttribute('data-action');
            
            // Remove selected class from all buttons
            protectionButtons.forEach(btn => btn.classList.remove('selected'));
            
            // Add selected class to clicked button
            this.classList.add('selected');
            
            // Handle defense indicator visibility
            const defenseIndicator = document.getElementById('defense-indicator');
            if (defenseIndicator) {
                if (action === 'run') {
                    defenseIndicator.classList.add('active');
                    console.log('Defense indicator activated');
                } else if (action === 'stop') {
                    defenseIndicator.classList.remove('active');
                    console.log('Defense indicator deactivated');
                }
            }
            
            // Send action to backend
            console.log('Protection action:', action);
            // Here you would typically make an API call to your backend
            // fetch('/api/protection', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify({ action })
            // });
        });
    });
});

// Function to update metric values and colors
function updateMetricBox(id, value) {
    const box = document.getElementById(id);
    if (!box) return;
    
    const valueElement = box.querySelector('.metric-value');
    if (valueElement) {
        valueElement.textContent = value + '%';
    }
    
    // Determine state based on value and metric type
    let state = '';
    if (id === 'age-protection' || id === 'gender-protection') {
        if (value < 35) state = 'low';
        else if (value < 70) state = 'medium';
        else state = 'high';
    } else if (id === 'age-leakage') {
        if (value < 35) state = 'low-inverse';
        else if (value < 70) state = 'medium-inverse';
        else state = 'high-inverse';
    } else if (id === 'gender-leakage') {
        if (value < 55) state = 'low-inverse';
        else if (value < 70) state = 'medium-inverse';
        else state = 'high-inverse';
    }
    
    // Update box state
    box.setAttribute('data-state', state);
}

// Example function to update all metrics (call this when receiving data from backend)
function updateAllMetrics(data) {
    if (data.ageProtection !== undefined) {
        updateMetricBox('age-protection', data.ageProtection);
    }
    if (data.genderProtection !== undefined) {
        updateMetricBox('gender-protection', data.genderProtection);
    }
    if (data.ageLeakage !== undefined) {
        updateMetricBox('age-leakage', data.ageLeakage);
    }
    if (data.genderLeakage !== undefined) {
        updateMetricBox('gender-leakage', data.genderLeakage);
    }
}

// Example of how to use the update functions:
// updateAllMetrics({
//     ageProtection: 75,
//     genderProtection: 82,
//     ageLeakage: 25,
//     genderLeakage: 45
// }); 

// Architecture Visualization Functionality
function initializeArchitectureVisualization() {
    console.log('Initializing architecture visualization...');
    
    const components = document.querySelectorAll('.draggable-component');
    const svg = document.querySelector('.connections-svg');
    const container = document.querySelector('.architecture-background');
    
    if (!svg || !container) {
        console.log('Architecture visualization elements not found');
        return;
    }
    
    let draggedElement = null;
    let offset = { x: 0, y: 0 };
    
    // Component positions for connection calculations
    const componentPositions = {};
    
    // Initialize component positions
    function updateComponentPositions() {
        let validPositions = 0;
        
        components.forEach(component => {
            const rect = component.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            // Check if component has valid positioning
            if (rect.width > 0 && rect.height > 0) {
                componentPositions[component.id] = {
                    x: rect.left - containerRect.left + rect.width / 2,
                    y: rect.top - containerRect.top + rect.height / 2
                };
                validPositions++;
            }
        });
        
        // Only update connection lines if we have valid positions
        if (validPositions === components.length) {
            updateConnectionLines();
        } else {
            // Retry after a short delay if not all components are positioned
            setTimeout(updateComponentPositions, 50);
        }
    }
    
    // Update connection lines based on component positions
    function updateConnectionLines() {
        const serverPos = componentPositions['server-component'];
        const agePos = componentPositions['age-inference'];
        const genderPos = componentPositions['gender-inference'];
        const imagePos = componentPositions['image-client'];
        const tabularPos = componentPositions['tabular-client'];
        
        if (!serverPos) return;
        
        // Update each connection line pair (send and receive)
        if (agePos) {
            updateLine('server-age-line-send', serverPos, agePos);
            updateLine('server-age-line-receive', serverPos, agePos);
        }
        if (genderPos) {
            updateLine('server-gender-line-send', serverPos, genderPos);
            updateLine('server-gender-line-receive', serverPos, genderPos);
        }
        if (imagePos) {
            updateLine('server-image-line-send', serverPos, imagePos);
            updateLine('server-image-line-receive', serverPos, imagePos);
        }
        if (tabularPos) {
            updateLine('server-tabular-line-send', serverPos, tabularPos);
            updateLine('server-tabular-line-receive', serverPos, tabularPos);
        }
    }
    
    // Update individual line coordinates with proper perpendicular offset
    function updateLine(lineId, startPos, endPos) {
        const line = document.getElementById(lineId);
        if (line) {
            // Calculate the direction vector
            const dx = endPos.x - startPos.x;
            const dy = endPos.y - startPos.y;
            const length = Math.sqrt(dx * dx + dy * dy);
            
            // Calculate perpendicular offset (3px spacing)
            const offsetDistance = 5;
            const perpX = (-dy / length) * offsetDistance;
            const perpY = (dx / length) * offsetDistance;
            
            // Apply offset based on line type (send vs receive)
            let offsetX = 0;
            let offsetY = 0;
            
            if (lineId.includes('-send')) {
                // Send lines get positive offset
                offsetX = perpX;
                offsetY = perpY;
            } else if (lineId.includes('-receive')) {
                // Receive lines get negative offset (opposite direction)
                offsetX = -perpX;
                offsetY = -perpY;
            }
            
            line.setAttribute('x1', startPos.x + offsetX);
            line.setAttribute('y1', startPos.y + offsetY);
            line.setAttribute('x2', endPos.x + offsetX);
            line.setAttribute('y2', endPos.y + offsetY);
        }
    }
    
    // Mouse down event handler
    function handleMouseDown(e) {
        if (e.target.closest('.draggable-component')) {
            draggedElement = e.target.closest('.draggable-component');
            draggedElement.classList.add('dragging');
            
            // Pause connection line animations during dragging
            document.querySelectorAll('.connection-line').forEach(line => {
                line.style.animationPlayState = 'paused';
            });
            
            const rect = draggedElement.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            offset.x = e.clientX - rect.left;
            offset.y = e.clientY - rect.top;
            
            e.preventDefault();
            console.log('Started dragging:', draggedElement.id);
        }
    }
    
    // Mouse move event handler
    function handleMouseMove(e) {
        if (!draggedElement) return;
        
        const containerRect = container.getBoundingClientRect();
        let newX = e.clientX - containerRect.left - offset.x;
        let newY = e.clientY - containerRect.top - offset.y;
        
        // Boundary checking
        const componentWidth = draggedElement.offsetWidth;
        const componentHeight = draggedElement.offsetHeight;
        
        newX = Math.max(0, Math.min(newX, container.offsetWidth - componentWidth));
        newY = Math.max(0, Math.min(newY, container.offsetHeight - componentHeight));
        
        draggedElement.style.left = newX + 'px';
        draggedElement.style.top = newY + 'px';
        draggedElement.style.transform = 'none'; // Remove initial transform
        
        // Use requestAnimationFrame for smoother updates
        requestAnimationFrame(updateComponentPositions);
    }
    
    // Mouse up event handler
    function handleMouseUp(e) {
        if (draggedElement) {
            draggedElement.classList.remove('dragging');
            
            // Resume connection line animations after dragging
            document.querySelectorAll('.connection-line').forEach(line => {
                line.style.animationPlayState = 'running';
            });
            
            console.log('Stopped dragging:', draggedElement.id);
            draggedElement = null;
        }
    }
    
    // Add event listeners
    container.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    
    // Initialize positions and connections with multiple attempts
    const initializeConnections = () => {
        updateComponentPositions();
        // Double-check after a short delay to ensure proper initialization
        setTimeout(() => {
            updateComponentPositions();
        }, 200);
    };
    
    // Initial setup
    setTimeout(initializeConnections, 100);
    
    // Backup initialization on window load
    window.addEventListener('load', () => {
        setTimeout(initializeConnections, 100);
    });
    
    // Update connections on window resize
    window.addEventListener('resize', () => {
        setTimeout(updateComponentPositions, 100);
    });
    
    console.log('Architecture visualization initialized');
}

// Add architecture visualization to existing DOMContentLoaded
// (This will be called from the main DOMContentLoaded function)

// Socket.IO initialization and event handlers
function initializeSocketIO() {
    socket = io();
    console.log('🔌 Socket.IO connecting...');
    
    socket.on('connect', function() {
        console.log('✅ Connected to dashboard server');
    });
    
    socket.on('disconnect', function() {
        console.log('❌ Disconnected from dashboard server');
    });
    
    // Configuration update confirmation
    socket.on('config_updated', function(data) {
        console.log('⚙️ Configuration updated:', data);
        // Update UI to reflect current config if needed
    });
    
    // Configuration error handling
    socket.on('config_error', function(data) {
        console.error('❌ Configuration error:', data.message);
        alert('Configuration Error: ' + data.message);
    });
    
    // Training status updates
    socket.on('training_status', function(data) {
        console.log('🎮 Training status:', data);
        
        const statusMessages = {
            'starting': '🚀 Starting FL training...',
            'stopped': '🛑 Training stopped',
            'error': '❌ Error: ' + data.message,
            'already_running': '⚠️ Training already running',
            'already_stopped': '⚠️ Training not running'
        };
        
        const message = statusMessages[data.status] || data.message;
        
        // Update UI status indicator
        const statusMessage = document.getElementById('current-status-message');
        if (statusMessage) {
            statusMessage.textContent = message;
        }
        
        // Update timer based on server response
        if (data.timer_running !== undefined) {
            console.log('🔄 Setting timer from training status:', data.timer_elapsed, data.timer_running);
            setTimerFromServer(data.timer_elapsed || 0, data.timer_running);
        }
        
        // Update progress if starting
        if (data.status === 'starting') {
            updateProgress(0);
        }
    });
    
    // Status message updates
    socket.on('status_update', function(data) {
        console.log('📋 Status update:', data);
        
        const statusMessage = document.getElementById('current-status-message');
        if (statusMessage && data.status) {
            statusMessage.textContent = data.status;
        }
    });
    
    // Detailed status updates
    socket.on('detailed_status_update', function(data) {
        console.log('📝 Detailed status update:', data);
        
        const statusMessage = document.getElementById('current-status-message');
        if (statusMessage && data.detailed_status) {
            statusMessage.textContent = data.detailed_status;
        }
    });
    
    // FL training data updates
    socket.on('fl_data_update', function(data) {
        console.log('📊 FL data update:', data);
        updateChartsWithLiveData(data);
        updateDashboardFromState(data);
        
        // Update timer from server data
        if (data.timer_elapsed !== undefined && data.timer_running !== undefined) {
            setTimerFromServer(data.timer_elapsed, data.timer_running);
        }
    });
    
    // Dashboard state updates (for compatibility)
    socket.on('dashboard_state', function(data) {
        console.log('📊 Dashboard state update:', data);
        updateDashboardFromState(data);
        updateChartsWithLiveData(data);
        
        // Update timer from initial state
        if (data.timer_elapsed !== undefined && data.timer_running !== undefined) {
            console.log('🔄 Setting timer from dashboard state:', data.timer_elapsed, data.timer_running);
            setTimerFromServer(data.timer_elapsed, data.timer_running);
        }
    });
    
    // Live updates (for compatibility)
    socket.on('live_update', function(data) {
        console.log('📈 Live update received:', data);
        updateChartsWithLiveData(data);
    });
    
    // Fairness data updates
    socket.on('fairness_update', function(data) {
        console.log('⚖️ Fairness update:', data);
        updateGenderFairnessChart(data.gender_fairness);
        updateAgeFairnessChart(data.age_fairness);
    });
    
    // Leakage data updates
    socket.on('leakage_update', function(data) {
        console.log('🔓 Leakage update:', data);
        updateGenderLeakageChart(data.gender_leakage);
        updateAgeLeakageChart(data.age_leakage);
    });
}

// Initialize charts
function initializeCharts() {
    // Register custom plugin for empty chart message
    Chart.register({
        id: 'emptyChart',
        afterDraw: function(chart) {
            if (chart.data.datasets[0].data.length === 0) {
                console.log('📊 Empty chart detected, showing waiting message for:', chart.canvas.id);
                const ctx = chart.ctx;
                const width = chart.width;
                const height = chart.height;
                
                ctx.save();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.font = '16px Montserrat';
                ctx.fillStyle = '#a0a0b8';
                ctx.fillText('Waiting for FL training data...', width / 2, height / 2);
                ctx.restore();
            }
        }
    });
    
    initializeAccuracyChart();
    initializeLossChart();
    initializeF1Chart();
    initializePrecisionRecallChart();
    initializeDefenseStrengthChart();
    initializeGenderFairnessChart();
    initializeAgeFairnessChart();
    initializeGenderLeakageChart();
    initializeAgeLeakageChart();
}

// Initialize accuracy chart
function initializeAccuracyChart() {
    const ctx = document.getElementById('accuracy-chart');
    if (!ctx) {
        console.log('❌ Accuracy chart canvas not found');
        return;
    }
    
    console.log('🔄 Initializing accuracy chart...');
    
    charts.accuracy = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Accuracy (%)',
                data: [],
                borderColor: '#8BA278',
                backgroundColor: 'rgba(139, 162, 120, 0.3)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#8BA278',
                pointBorderColor: '#FFFFFD',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'FL Round',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
    console.log('✅ Accuracy chart initialized');
}

// Initialize loss chart
function initializeLossChart() {
    const ctx = document.getElementById('loss-chart');
    if (!ctx) return;
    
    charts.loss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#7A80F2',
                backgroundColor: 'rgba(122, 128, 242, 0.3)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#7A80F2',
                pointBorderColor: '#FFFFFD',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'FL Round',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            }
        }
    });
    console.log('✅ Loss chart initialized');
}

// Initialize F1 chart
function initializeF1Chart() {
    const ctx = document.getElementById('f1-chart');
    if (!ctx) return;
    
    charts.f1 = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'F1 Score (%)',
                data: [],
                borderColor: '#DDBA5E',
                backgroundColor: 'rgba(221, 186, 94, 0.3)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#DDBA5E',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'FL Round',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'F1 Score (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            }
        }
    });
    console.log('✅ F1 chart initialized');
}

// Initialize Precision-Recall chart
function initializePrecisionRecallChart() {
    const ctx = document.getElementById('precision-recall-chart');
    if (!ctx) return;
    
    charts.precisionRecall = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Precision-Recall (%)',
                data: [],
                borderColor: '#EE8945',
                backgroundColor: 'rgba(238, 137, 69, 0.3)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#EE8945',
                pointBorderColor: '#FFFFFD',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'FL Round',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precision-Recall (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            }
        }
    });
    console.log('✅ Precision-Recall chart initialized');
}

// Initialize Defense Strength chart
function initializeDefenseStrengthChart() {
    const ctx = document.getElementById('defense-strength-chart');
    if (!ctx) return;
    
    charts.defenseStrength = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Defense Strength (%)',
                data: [],
                borderColor: '#8BA278',
                backgroundColor: 'rgba(139, 162, 120, 0.3)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#8BA278',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'FL Round',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Defense Strength (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            }
        }
    });
    console.log('✅ Defense Strength chart initialized');
}

// Initialize gender fairness chart (horizontal bar chart)
function initializeGenderFairnessChart() {
    const ctx = document.getElementById('gender-fairness-chart');
    if (!ctx) {
        console.log('❌ Gender fairness chart canvas not found');
        return;
    }
    
    console.log('🔄 Initializing gender fairness chart...');
    
    charts.genderFairness = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Female', 'Male'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [0, 0], // Start with 0 values
                backgroundColor: ['#F7D78E', '#FFA569'],
                borderColor: ['#F7D78E', '#FFA569'],
                borderWidth: 1,
                borderRadius: 20,
                borderSkipped: false
            }]
        },
        options: {
            indexAxis: 'y', // Horizontal bars
            responsive: true,
            maintainAspectRatio: false,
            // Bar width controls
            categoryPercentage: 0.8, // Controls spacing between categories (0.1-1.0)
            barPercentage: 0.6,      // Controls bar thickness within category (0.1-1.0)
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
    console.log('✅ Gender fairness chart initialized');
}

// Initialize age fairness chart (vertical bar chart)
function initializeAgeFairnessChart() {
    const ctx = document.getElementById('age-fairness-chart');
    if (!ctx) {
        console.log('❌ Age fairness chart canvas not found');
        return;
    }
    
    console.log('🔄 Initializing age fairness chart...');
    
    charts.ageFairness = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['-30', '31-40', '41-50', '51-60', '61-70', '+71'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [0, 0, 0, 0, 0, 0], // Start with 0 values
                backgroundColor: '#F7D78E',
                borderColor: '#F7D78E',
                borderWidth: 1,
                borderRadius: 20,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            // Bar width controls  
            categoryPercentage: 0.6, // Controls spacing between categories (0.1-1.0)
            barPercentage: 0.5,      // Controls bar thickness within category (0.1-1.0)
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Age Generation',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Accuracy (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
    console.log('✅ Age fairness chart initialized');
}

// Initialize gender leakage chart (line chart)
function initializeGenderLeakageChart() {
    const ctx = document.getElementById('gender-leakage-chart');
    if (!ctx) {
        console.log('❌ Gender leakage chart canvas not found');
        return;
    }
    
    console.log('🔄 Initializing gender leakage chart...');
    
    charts.genderLeakage = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Gender Leakage (%)',
                data: [],
                borderColor: '#EE8945',
                backgroundColor: 'rgba(238, 137, 69, 0.3)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#EE8945',
                pointBorderColor: '#FFFFFD',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'FL Round',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Leakage Accuracy (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
    console.log('✅ Gender leakage chart initialized');
}

// Initialize age leakage chart (line chart)
function initializeAgeLeakageChart() {
    const ctx = document.getElementById('age-leakage-chart');
    if (!ctx) {
        console.log('❌ Age leakage chart canvas not found');
        return;
    }
    
    console.log('🔄 Initializing age leakage chart...');
    
    charts.ageLeakage = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Age Leakage (%)',
                data: [],
                borderColor: '#F7D78E',
                backgroundColor: 'rgba(247, 215, 142, 0.3)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#F7D78E',
                pointBorderColor: '#FFFFFD',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'FL Round',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Leakage Accuracy (%)',
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    },
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: '#CCCCCC'
                    },
                    ticks: {
                        color: '#808080',
                        font: {
                            family: 'Montserrat',
                            weight: '400',
                            size: 12
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
    console.log('✅ Age leakage chart initialized');
}

// Update dashboard from state
function updateDashboardFromState(data) {
    if (data.config) {
        // Update round info
        const currentRoundEl = document.querySelector('.current-round');
        const totalRoundsEl = document.querySelector('.total-rounds');
        if (currentRoundEl) currentRoundEl.textContent = data.config.current_round || 0;
        if (totalRoundsEl) totalRoundsEl.textContent = data.config.federated_rounds || 2;
        
        // Update progress
        if (data.config.progress !== undefined) {
            updateProgress(data.config.progress);
        }
    }
    
    if (data.metrics && data.metrics.home) {
        // Update status
        const statusEl = document.querySelector('.training-status');
        const statusLine2El = document.querySelector('.status-line2');
        if (statusEl) statusEl.textContent = data.metrics.home.training_status || '';
        if (statusLine2El) statusLine2El.textContent = data.metrics.home.status_line2 || '';
    }
}

// Update charts with live FL data
function updateChartsWithLiveData(data) {
    console.log('🔄 Updating charts with FL data:', data);
    
    // Add detailed debugging
    if (data.metrics && data.metrics.performance) {
        const performance = data.metrics.performance;
        console.log('📊 Performance data received:', performance);
        console.log('📊 Performance arrays - Accuracy:', performance.live_accuracy, 'Loss:', performance.live_loss, 'F1:', performance.f1_score, 'P-R:', performance.precision_recall);
        console.log('📈 Live accuracy array:', performance.live_accuracy);
        console.log('📉 Live loss array:', performance.live_loss);
        console.log('🎯 F1 score array:', performance.f1_score);
        
        // Update accuracy chart
        if (charts.accuracy) {
            if (performance.live_accuracy && performance.live_accuracy.length > 0) {
                const rounds = performance.live_accuracy.map((_, index) => `Round ${index + 1}`);
                console.log('🔄 Updating accuracy chart with rounds:', rounds, 'data:', performance.live_accuracy);
                charts.accuracy.data.labels = rounds;
                charts.accuracy.data.datasets[0].data = performance.live_accuracy;
                charts.accuracy.update('none');
                console.log('✅ Accuracy chart updated successfully');
            } else {
                // Clear chart and show waiting message
                charts.accuracy.data.labels = [];
                charts.accuracy.data.datasets[0].data = [];
                charts.accuracy.update('none');
                console.log('🔄 Accuracy chart cleared - waiting for data');
            }
        } else {
            console.log('❌ Accuracy chart not initialized');
        }
        
        // Update loss chart
        if (charts.loss) {
            if (performance.live_loss && performance.live_loss.length > 0) {
                const rounds = performance.live_loss.map((_, index) => `Round ${index + 1}`);
                console.log('🔄 Updating loss chart with rounds:', rounds, 'data:', performance.live_loss);
                charts.loss.data.labels = rounds;
                charts.loss.data.datasets[0].data = performance.live_loss;
                charts.loss.update('none');
                console.log('✅ Loss chart updated successfully');
            } else {
                // Clear chart and show waiting message
                charts.loss.data.labels = [];
                charts.loss.data.datasets[0].data = [];
                charts.loss.update('none');
                console.log('🔄 Loss chart cleared - waiting for data');
            }
        } else {
            console.log('❌ Loss chart not initialized');
        }
        
        // Update F1 chart
        if (charts.f1) {
            if (performance.f1_score && performance.f1_score.length > 0) {
                const rounds = performance.f1_score.map((_, index) => `Round ${index + 1}`);
                console.log('🔄 Updating F1 chart with rounds:', rounds, 'data:', performance.f1_score);
                charts.f1.data.labels = rounds;
                charts.f1.data.datasets[0].data = performance.f1_score;
                charts.f1.update('none');
                console.log('✅ F1 chart updated successfully');
            } else {
                // Clear chart and show waiting message
                charts.f1.data.labels = [];
                charts.f1.data.datasets[0].data = [];
                charts.f1.update('none');
                console.log('🔄 F1 chart cleared - waiting for data');
            }
        } else {
            console.log('❌ F1 chart not initialized');
        }
        
        // Update Precision-Recall chart
        if (charts.precisionRecall) {
            if (performance.precision_recall && performance.precision_recall.length > 0) {
                const rounds = performance.precision_recall.map((_, index) => `Round ${index + 1}`);
                console.log('🔄 Updating Precision-Recall chart with rounds:', rounds, 'data:', performance.precision_recall);
                charts.precisionRecall.data.labels = rounds;
                charts.precisionRecall.data.datasets[0].data = performance.precision_recall;
                charts.precisionRecall.update('none');
                console.log('✅ Precision-Recall chart updated successfully');
            } else {
                // Clear chart and show waiting message
                charts.precisionRecall.data.labels = [];
                charts.precisionRecall.data.datasets[0].data = [];
                charts.precisionRecall.update('none');
                console.log('🔄 Precision-Recall chart cleared - waiting for data');
            }
        } else {
            console.log('❌ Precision-Recall chart not initialized');
        }
    } else {
        console.log('❌ No performance metrics in data:', data);
    }
    
    // Update Defense Strength chart
    if (data.metrics && data.metrics.defence) {
        const defence = data.metrics.defence;
        console.log('🛡️ Defense data received:', defence);
        console.log('🛡️ Defense strength array:', defence.defence_strength);
        
        if (charts.defenseStrength) {
            if (defence.defence_strength && defence.defence_strength.length > 0) {
                const rounds = defence.defence_strength.map((_, index) => `Round ${index + 1}`);
                console.log('🔄 Updating Defense Strength chart with rounds:', rounds, 'data:', defence.defence_strength);
                charts.defenseStrength.data.labels = rounds;
                charts.defenseStrength.data.datasets[0].data = defence.defence_strength;
                charts.defenseStrength.update('none');
                console.log('✅ Defense Strength chart updated successfully');
            } else {
                // Clear chart and show waiting message
                charts.defenseStrength.data.labels = [];
                charts.defenseStrength.data.datasets[0].data = [];
                charts.defenseStrength.update('none');
                console.log('🔄 Defense Strength chart cleared - waiting for data');
            }
        } else {
            console.log('❌ Defense Strength chart not initialized');
        }
    } else {
        console.log('❌ No defense metrics in data:', data);
    }
    
    // Update Gender Fairness chart
    if (data.metrics && data.metrics.performance && data.metrics.performance.gender_accuracy) {
        const genderData = data.metrics.performance.gender_accuracy;
        console.log('🚺🚹 Gender fairness data received:', genderData);
        
        if (charts.genderFairness) {
            console.log('🔄 Updating Gender Fairness chart with data:', genderData);
            charts.genderFairness.data.datasets[0].data = genderData;
            charts.genderFairness.update('none');
            console.log('✅ Gender Fairness chart updated successfully');
        } else {
            console.log('❌ Gender Fairness chart not initialized');
        }
    }
    
    // Update Age Fairness chart
    if (data.metrics && data.metrics.performance && data.metrics.performance.age_accuracy) {
        const ageData = data.metrics.performance.age_accuracy;
        console.log('👶👴 Age fairness data received:', ageData);
        
        if (charts.ageFairness) {
            console.log('🔄 Updating Age Fairness chart with data:', ageData);
            charts.ageFairness.data.datasets[0].data = ageData;
            charts.ageFairness.update('none');
            console.log('✅ Age Fairness chart updated successfully');
        } else {
            console.log('❌ Age Fairness chart not initialized');
        }
    }
    
    // Update Gender Leakage chart
    if (data.metrics && data.metrics.attack) {
        const attack = data.metrics.attack;
        console.log('🎯 Attack data received:', attack);
        
        if (charts.genderLeakage) {
            if (attack.gender_leakage && attack.gender_leakage.length > 0) {
                const rounds = attack.gender_leakage.map((_, index) => `Round ${index + 1}`);
                console.log('🔄 Updating Gender Leakage chart with rounds:', rounds, 'data:', attack.gender_leakage);
                charts.genderLeakage.data.labels = rounds;
                charts.genderLeakage.data.datasets[0].data = attack.gender_leakage;
                charts.genderLeakage.update('none');
                console.log('✅ Gender Leakage chart updated successfully');
            } else {
                // Clear chart and show waiting message
                charts.genderLeakage.data.labels = [];
                charts.genderLeakage.data.datasets[0].data = [];
                charts.genderLeakage.update('none');
                console.log('🔄 Gender Leakage chart cleared - waiting for data');
            }
        } else {
            console.log('❌ Gender Leakage chart not initialized');
        }
        
        // Update Age Leakage chart
        if (charts.ageLeakage) {
            if (attack.age_leakage && attack.age_leakage.length > 0) {
                const rounds = attack.age_leakage.map((_, index) => `Round ${index + 1}`);
                console.log('🔄 Updating Age Leakage chart with rounds:', rounds, 'data:', attack.age_leakage);
                charts.ageLeakage.data.labels = rounds;
                charts.ageLeakage.data.datasets[0].data = attack.age_leakage;
                charts.ageLeakage.update('none');
                console.log('✅ Age Leakage chart updated successfully');
            } else {
                // Clear chart and show waiting message
                charts.ageLeakage.data.labels = [];
                charts.ageLeakage.data.datasets[0].data = [];
                charts.ageLeakage.update('none');
                console.log('🔄 Age Leakage chart cleared - waiting for data');
            }
        } else {
            console.log('❌ Age Leakage chart not initialized');
        }
    } else {
        console.log('❌ No attack metrics in data:', data);
    }
} 