# 🚀 Streamlit Deployment Guide

This guide will help you deploy the Recommendation System on Streamlit for web-based access.

## 📋 Prerequisites

- Python 3.8 or higher
- All dependencies from `requirements.txt`
- Cleaned dataset files (events_cleaned.csv, item_properties_cleaned.csv, category_tree_cleaned.csv)

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Data Files

Ensure you have the following cleaned data files in your project directory:
- `events_cleaned.csv`
- `item_properties_cleaned.csv` 
- `category_tree_cleaned.csv`

If you don't have cleaned files, the app will automatically clean the raw data files.

## 🚀 Running the Application

### Method 1: Direct Streamlit Command

```bash
streamlit run streamlit_app.py
```

### Method 2: Using the Deployment Script

```bash
python run_streamlit.py
```

### Method 3: Custom Configuration

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 🌐 Accessing the Application

Once running, the application will be available at:
- **Local**: http://localhost:8501
- **Network**: http://0.0.0.0:8501 (accessible from other devices on the network)

## 📱 Application Features

### 🏠 Home Page
- System overview and quick statistics
- Data loading interface
- System health status

### 🎯 Recommendations Page
- **User Selection**: Choose from available users in the system
- **Algorithm Selection**: 
  - Hybrid (recommended)
  - Collaborative Filtering
  - Content-Based
  - Matrix Factorization
  - Popularity-Based
- **Customization**: Adjust number of recommendations (1-20)
- **Real-time Results**: Instant recommendation generation

### 📊 Analytics Dashboard
- **Event Distribution**: Visual breakdown of user interactions
- **Temporal Analysis**: Hourly and daily activity patterns
- **User Activity**: Behavior analysis and statistics
- **Item Popularity**: Most viewed and trending items

### ⚙️ System Status
- **Performance Metrics**: System health and statistics
- **Data Quality**: Matrix sparsity and interaction coverage
- **Configuration**: Current system parameters

### 🔧 Data Management
- **Data Cleaning**: Automated dataset preprocessing
- **File Status**: Check availability of required files
- **System Configuration**: View and modify parameters

## ⚡ Performance Optimization

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Data processing limits
export MAX_ROWS_EVENTS=300000
export MAX_ROWS_ITEM_PROPERTIES=200000
export TOP_USERS_LIMIT=1000
export TOP_ITEMS_LIMIT=1000
export COLLAB_SIMILAR_USERS=10

# Evaluation settings
export N_TEST_USERS=100
export EVAL_TOP_N=10
export EVAL_USER_LIMIT=50

# Performance tuning
export SKIP_HEAVY_ANALYTICS=1
```

### Memory Management

The application includes several memory optimization features:
- **Data Sampling**: Automatic sampling for large datasets
- **Caching**: Streamlit caching for expensive operations
- **Lazy Loading**: Data loaded only when needed

## 🔧 Configuration

### Streamlit Configuration

The application uses `.streamlit/config.toml` for configuration:

```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Custom Styling

The application includes custom CSS for enhanced user experience:
- Responsive design
- Professional color scheme
- Interactive components
- Mobile-friendly layout

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `MAX_ROWS_EVENTS` and `MAX_ROWS_ITEM_PROPERTIES`
   - Enable `SKIP_HEAVY_ANALYTICS=1`

2. **Data Loading Issues**
   - Ensure cleaned CSV files exist
   - Check file permissions
   - Verify CSV format and encoding

3. **Performance Issues**
   - Reduce `TOP_USERS_LIMIT` and `TOP_ITEMS_LIMIT`
   - Use smaller datasets for testing
   - Enable caching with `@st.cache_data`

### Debug Mode

Run with debug information:
```bash
streamlit run streamlit_app.py --logger.level debug
```

## 📊 Monitoring

### System Metrics

The application provides real-time monitoring of:
- **Data Quality**: Matrix sparsity, user coverage
- **Performance**: Loading times, recommendation speed
- **Usage**: User interactions, system health

### Logs

Streamlit logs are available in:
- Console output
- Streamlit's built-in logging
- Browser developer tools

## 🚀 Production Deployment

### For Production Use

1. **Security**: Configure proper authentication
2. **Scaling**: Use multiple workers for high traffic
3. **Monitoring**: Set up proper logging and monitoring
4. **Backup**: Regular data backups

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📈 Scaling Considerations

- **Horizontal Scaling**: Multiple Streamlit instances
- **Data Caching**: Redis or similar for shared caching
- **Database**: Move from CSV to proper database
- **API**: Consider FastAPI for high-performance API

## 🆘 Support

For issues and questions:
1. Check the application logs
2. Verify data file integrity
3. Test with smaller datasets
4. Review system requirements

## 📝 Notes

- The application automatically handles data cleaning
- Caching is enabled for better performance
- All visualizations are interactive
- Mobile-responsive design included
- Real-time recommendation generation
