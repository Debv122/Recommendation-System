#!/usr/bin/env python3
"""
Test script for Streamlit deployment
Verifies that all components are working correctly
"""

import sys
import os
import importlib.util

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'plotly',
        'matplotlib',
        'seaborn'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            if module == 'sklearn':
                import sklearn
            else:
                __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful!")
    return True

def test_recommendation_system():
    """Test if the recommendation system can be imported"""
    print("\n🔍 Testing recommendation system...")
    
    try:
        from Recommendation_system import RecommendationSystem
        print("✅ RecommendationSystem imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import RecommendationSystem: {e}")
        return False

def test_data_files():
    """Test if required data files exist"""
    print("\n🔍 Testing data files...")
    
    required_files = [
        'events_cleaned.csv',
        'item_properties_cleaned.csv',
        'category_tree_cleaned.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️ {file} not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        print("The app will attempt to clean raw data files automatically.")
        return True  # Not critical, app can handle this
    
    print("✅ All data files found!")
    return True

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    print("\n🔍 Testing Streamlit app...")
    
    try:
        # Try to import the app module
        spec = importlib.util.spec_from_file_location("streamlit_app", "streamlit_app.py")
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        print("✅ Streamlit app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import Streamlit app: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Streamlit Deployment")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_recommendation_system,
        test_data_files,
        test_streamlit_app
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All tests passed! ({passed}/{total})")
        print("\n🚀 Ready to run Streamlit app!")
        print("Run: streamlit run streamlit_app.py")
        return True
    else:
        print(f"❌ Some tests failed ({passed}/{total})")
        print("Please fix the issues above before running the app.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
