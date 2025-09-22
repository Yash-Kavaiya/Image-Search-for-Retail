// Image Search JavaScript
class ImageSearchApp {
    constructor() {
        this.currentTab = 'image-search';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDatabaseStats();
        console.log('Image Search App initialized');
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.search-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.id.replace('-tab', '')));
        });

        // File inputs
        document.getElementById('image-file').addEventListener('change', (e) => this.handleImageUpload(e, 'image'));
        document.getElementById('multimodal-image-file').addEventListener('change', (e) => this.handleImageUpload(e, 'multimodal'));

        // Search buttons
        document.getElementById('image-search-btn').addEventListener('click', () => this.performImageSearch());
        document.getElementById('text-search-btn').addEventListener('click', () => this.performTextSearch());
        document.getElementById('multimodal-search-btn').addEventListener('click', () => this.performMultimodalSearch());

        // Database initialization
        document.getElementById('initialize-db-btn').addEventListener('click', () => this.initializeDatabase());

        // Drag and drop
        this.setupDragAndDrop();
    }

    switchTab(tabName) {
        // Update tab styles
        document.querySelectorAll('.search-tab').forEach(tab => {
            tab.classList.remove('active', 'border-primary', 'text-primary');
            tab.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');
        });

        document.getElementById(`${tabName}-tab`).classList.add('active', 'border-primary', 'text-primary');
        document.getElementById(`${tabName}-tab`).classList.remove('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300');

        // Show/hide panels
        document.querySelectorAll('.search-panel').forEach(panel => {
            panel.classList.add('hidden');
        });

        document.getElementById(`${tabName}-panel`).classList.remove('hidden');
        this.currentTab = tabName;
    }

    setupDragAndDrop() {
        const dropZones = document.querySelectorAll('[class*="border-dashed"]');
        
        dropZones.forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('border-primary', 'bg-primary', 'bg-opacity-5');
            });

            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('border-primary', 'bg-primary', 'bg-opacity-5');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('border-primary', 'bg-primary', 'bg-opacity-5');
                
                const files = e.dataTransfer.files;
                if (files.length > 0 && files[0].type.startsWith('image/')) {
                    const isMultimodal = zone.closest('#multimodal-search-panel');
                    const input = isMultimodal ? 
                        document.getElementById('multimodal-image-file') : 
                        document.getElementById('image-file');
                    input.files = files;
                    this.handleImageUpload({ target: input }, isMultimodal ? 'multimodal' : 'image');
                }
            });
        });
    }

    handleImageUpload(event, type) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('Image file must be less than 10MB.');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewId = type === 'multimodal' ? 'multimodal-image-preview' : 'image-preview';
            const imgId = type === 'multimodal' ? 'multimodal-preview-img' : 'preview-img';
            
            document.getElementById(previewId).classList.remove('hidden');
            document.getElementById(imgId).src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Enable search button
        const buttonId = type === 'multimodal' ? 'multimodal-search-btn' : 'image-search-btn';
        document.getElementById(buttonId).disabled = false;
    }

    async performImageSearch() {
        const fileInput = document.getElementById('image-file');
        const file = fileInput.files[0];

        if (!file) {
            this.showError('Please select an image first.');
            return;
        }

        this.showLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('top_k', '4');

            const response = await fetch('/search/image', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result.results, 'Image Search');
            } else {
                this.showError('Search failed. Please try again.');
            }
        } catch (error) {
            console.error('Image search error:', error);
            this.showError('Search failed. Please check your connection and try again.');
        } finally {
            this.showLoading(false);
        }
    }

    async performTextSearch() {
        const query = document.getElementById('text-query').value.trim();

        if (!query) {
            this.showError('Please enter a search query.');
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch('/search/text?top_k=4', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query })
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result.results, `Text Search: "${query}"`);
            } else {
                this.showError('Search failed. Please try again.');
            }
        } catch (error) {
            console.error('Text search error:', error);
            this.showError('Search failed. Please check your connection and try again.');
        } finally {
            this.showLoading(false);
        }
    }

    async performMultimodalSearch() {
        const fileInput = document.getElementById('multimodal-image-file');
        const file = fileInput.files[0];
        const query = document.getElementById('multimodal-query').value.trim();

        if (!file) {
            this.showError('Please select an image first.');
            return;
        }

        if (!query) {
            this.showError('Please add a text description.');
            return;
        }

        this.showLoading(true);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('query', query);
            formData.append('top_k', '4');

            const response = await fetch('/search/multimodal', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayResults(result.results, `Multimodal Search: "${query}"`);
            } else {
                this.showError('Search failed. Please try again.');
            }
        } catch (error) {
            console.error('Multimodal search error:', error);
            this.showError('Search failed. Please check your connection and try again.');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(results, searchType) {
        const resultsSection = document.getElementById('results-section');
        const resultsGrid = document.getElementById('results-grid');
        const resultsCount = document.getElementById('results-count');

        // Clear previous results
        resultsGrid.innerHTML = '';

        if (results.length === 0) {
            resultsGrid.innerHTML = `
                <div class="col-span-full text-center py-8 text-gray-500">
                    <i class="fas fa-search text-4xl mb-4"></i>
                    <p>No similar products found. Try a different search.</p>
                </div>
            `;
        } else {
            results.forEach(product => {
                const productCard = this.createProductCard(product);
                resultsGrid.appendChild(productCard);
            });
        }

        resultsCount.textContent = `${results.length} result${results.length !== 1 ? 's' : ''} found for ${searchType}`;
        resultsSection.classList.remove('hidden');

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    createProductCard(product) {
        const card = document.createElement('div');
        card.className = 'bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow';

        const price = product.sale_price > 0 ? `$${product.sale_price.toFixed(2)}` : 
                     product.list_price > 0 ? `$${product.list_price.toFixed(2)}` : 'Price not available';

        const similarityPercentage = (product.similarity_score * 100).toFixed(1);
        // Determine thumbnail URL or generate placeholder SVG
        let thumbUrl = '';
        if (product.image_url && product.image_url.trim() !== '') {
            thumbUrl = product.image_url;
        } else {
            // Create a small SVG placeholder with initials
            const initials = (product.product_name || 'P').split(' ').slice(0,2).map(s => s[0] || '').join('').toUpperCase();
            const svg = `<?xml version='1.0' encoding='utf-8'?><svg xmlns='http://www.w3.org/2000/svg' width='320' height='320'><rect width='100%' height='100%' fill='%23F3F4F6'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-family='Arial, Helvetica, sans-serif' font-size='48' fill='%239CA3AF'>${initials}</text></svg>`;
            thumbUrl = 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);
        }

        card.innerHTML = `
            <div class="h-40 w-full bg-gray-100 flex items-center justify-center overflow-hidden">
                <img src="${thumbUrl}" alt="${product.product_name}" class="object-cover w-full h-full">
            </div>
            <div class="p-4">
                <div class="flex justify-between items-start mb-2">
                    <h4 class="text-sm font-semibold text-gray-900 line-clamp-2">${product.product_name}</h4>
                    <span class="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full ml-2 whitespace-nowrap">
                        ${similarityPercentage}% match
                    </span>
                </div>
                <p class="text-xs text-gray-600 mb-2">${product.brand}</p>
                <p class="text-sm text-gray-700 line-clamp-3 mb-3">${product.description}</p>
                <div class="flex justify-between items-center">
                    <span class="text-lg font-bold text-primary">${price}</span>
                    ${product.product_url ? `
                        <a href="${product.product_url}" target="_blank" 
                           class="text-xs bg-primary text-white px-3 py-1 rounded hover:bg-primary-dark transition-colors">
                            View Product
                        </a>
                    ` : ''}
                </div>
                <div class="mt-2 text-xs text-gray-500">
                    <span class="inline-block bg-gray-100 rounded px-2 py-1">${product.category}</span>
                </div>
            </div>
        `;

        return card;
    }

    async loadDatabaseStats() {
        try {
            const response = await fetch('/products/stats');
            const result = await response.json();

            if (result.success) {
                const stats = result.database_stats;
                document.getElementById('total-products').textContent = stats.total_products;
                document.getElementById('products-with-embeddings').textContent = stats.products_with_embeddings;
                document.getElementById('unique-brands').textContent = stats.unique_brands;
                document.getElementById('unique-categories').textContent = stats.unique_categories;
            }
        } catch (error) {
            console.error('Failed to load database stats:', error);
        }
    }

    async initializeDatabase() {
        const button = document.getElementById('initialize-db-btn');
        const originalText = button.innerHTML;
        
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Initializing...';

        try {
            const response = await fetch('/products/initialize', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                this.showSuccess('Database initialized successfully!');
                this.loadDatabaseStats();
            } else {
                this.showError('Failed to initialize database.');
            }
        } catch (error) {
            console.error('Database initialization error:', error);
            this.showError('Failed to initialize database. Check console for details.');
        } finally {
            button.disabled = false;
            button.innerHTML = originalText;
        }
    }

    showLoading(show) {
        const loading = document.getElementById('loading');
        if (show) {
            loading.classList.remove('hidden');
        } else {
            loading.classList.add('hidden');
        }
    }

    showError(message) {
        // Simple alert for now - could be replaced with a toast notification
        alert('Error: ' + message);
    }

    showSuccess(message) {
        // Simple alert for now - could be replaced with a toast notification
        alert('Success: ' + message);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageSearchApp();
});

// Add utility CSS classes for line-clamp
const style = document.createElement('style');
style.textContent = `
    .line-clamp-2 {
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .line-clamp-3 {
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .search-tab.active {
        border-color: #4F46E5;
        color: #4F46E5;
    }
    
    .search-tab:not(.active) {
        border-color: transparent;
        color: #6B7280;
    }
    
    .search-tab:not(.active):hover {
        color: #374151;
        border-color: #D1D5DB;
    }
`;
document.head.appendChild(style);