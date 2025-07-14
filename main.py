import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Amazon Commerce Reviews Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">📊 Amazon Commerce Reviews Analysis Dashboard</div>', unsafe_allow_html=True)

# Sidebar for data loading and configuration
st.sidebar.header("🔧 Configuration")

# Data loading section
@st.cache_data
def load_sample_data():
    """Create sample data if no file is uploaded"""
    np.random.seed(42)
    n_samples = 1500
    
    # Create sample features that might represent Amazon review characteristics
    data = {
        'review_length': np.random.normal(150, 50, n_samples),
        'word_count': np.random.poisson(30, n_samples),
        'sentence_count': np.random.poisson(5, n_samples),
        'avg_word_length': np.random.normal(4.5, 1.2, n_samples),
        'punctuation_ratio': np.random.beta(2, 8, n_samples),
        'digit_ratio': np.random.beta(1, 20, n_samples),
        'uppercase_ratio': np.random.beta(1, 10, n_samples),
        'sentiment_score': np.random.normal(0.6, 0.3, n_samples)
    }
    
    # Create target variable (author_id)
    data['author_id'] = np.random.randint(1, 51, n_samples)
    
    return pd.DataFrame(data)

@st.cache_data
def load_uci_data():
    """Load data from UCI repository"""
    try:
        from ucimlrepo import fetch_ucirepo
        amazon_commerce_reviews = fetch_ucirepo(id=215)
        X = amazon_commerce_reviews.data.features
        y = amazon_commerce_reviews.data.targets
        
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading UCI data: {e}")
        return None

# Data loading options
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["Sample Data", "Upload CSV", "Load from UCI Repository"]
)

df = None

if data_source == "Sample Data":
    df = load_sample_data()
    st.sidebar.success("Sample data loaded successfully!")
    
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            
elif data_source == "Load from UCI Repository":
    if st.sidebar.button("Load UCI Data"):
        with st.spinner("Loading data from UCI repository..."):
            df = load_uci_data()
            if df is not None:
                st.sidebar.success("UCI data loaded successfully!")

# Main content
if df is not None:
    # Dataset overview
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        if 'author_id' in df.columns:
            st.metric("Unique Authors", df['author_id'].nunique())
        else:
            st.metric("Numeric Features", df.select_dtypes(include=[np.number]).shape[1])
    
    # Display basic statistics
    st.subheader("📊 Dataset Statistics")
    st.dataframe(df.describe())
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Data Exploration", "📈 Visualizations", "🔗 Correlations", "🤖 Machine Learning", "📄 Data Table"])
    
    with tab1:
        st.header("Data Exploration")
        
        # Feature selection for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Distribution")
                selected_feature = st.selectbox("Select a feature to analyze", numeric_columns)
                
                if selected_feature:
                    fig = px.histogram(df, x=selected_feature, nbins=30, 
                                     title=f"Distribution of {selected_feature}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Box Plot Analysis")
                if len(numeric_columns) > 1:
                    box_feature = st.selectbox("Select feature for box plot", numeric_columns, key="box")
                    
                    if box_feature:
                        fig = px.box(df, y=box_feature, title=f"Box Plot of {box_feature}")
                        st.plotly_chart(fig, use_container_width=True)
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        if df.isnull().sum().sum() > 0:
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        title="Missing Values by Feature")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    with tab2:
        st.header("Data Visualizations")
        
        if len(numeric_columns) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Scatter Plot")
                x_axis = st.selectbox("Select X-axis", numeric_columns, key="scatter_x")
                y_axis = st.selectbox("Select Y-axis", numeric_columns, key="scatter_y", index=1)
                
                if x_axis and y_axis:
                    fig = px.scatter(df, x=x_axis, y=y_axis, 
                                   title=f"{x_axis} vs {y_axis}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Feature Comparison")
                features_to_compare = st.multiselect("Select features to compare", 
                                                   numeric_columns, 
                                                   default=numeric_columns[:min(4, len(numeric_columns))])
                
                if features_to_compare:
                    # Normalize data for comparison
                    normalized_df = df[features_to_compare].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
                    
                    fig = go.Figure()
                    for feature in features_to_compare:
                        fig.add_trace(go.Scatter(
                            y=normalized_df[feature].values,
                            mode='lines',
                            name=feature,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(title="Normalized Feature Comparison",
                                    xaxis_title="Sample Index",
                                    yaxis_title="Normalized Value")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Pairplot for selected features
        st.subheader("Pairwise Feature Relationships")
        if len(numeric_columns) > 2:
            selected_features = st.multiselect("Select features for pairplot", 
                                             numeric_columns, 
                                             default=numeric_columns[:min(4, len(numeric_columns))])
            
            if len(selected_features) > 1:
                fig = px.scatter_matrix(df[selected_features], 
                                      title="Pairwise Scatter Plot Matrix")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Correlation Analysis")
        
        if len(numeric_columns) > 1:
            # Correlation heatmap
            correlation_matrix = df[numeric_columns].corr()
            
            fig = px.imshow(correlation_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale="RdBu",
                          title="Feature Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.subheader("Top Correlations")
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            st.dataframe(corr_df.head(10))
    
    with tab4:
        st.header("Machine Learning Analysis")
        
        if len(numeric_columns) > 1:
            # Feature and target selection
            col1, col2 = st.columns(2)
            
            with col1:
                target_column = st.selectbox("Select target variable", df.columns.tolist())
                
            with col2:
                feature_columns = st.multiselect("Select features", 
                                                numeric_columns, 
                                                default=[col for col in numeric_columns if col != target_column])
            
            if target_column and feature_columns:
                try:
                    X = df[feature_columns]
                    y = df[target_column]
                    
                    # Handle different target types
                    if y.dtype == 'object' or y.nunique() < 20:
                        # Classification
                        st.subheader("Classification Analysis")
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                        # Scaling
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Model selection
                        model_choice = st.selectbox("Select model", 
                                                   ["Random Forest", "Logistic Regression", "SVM"])
                        
                        if st.button("Train Model"):
                            with st.spinner("Training model..."):
                                if model_choice == "Random Forest":
                                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                                    model.fit(X_train, y_train)
                                elif model_choice == "Logistic Regression":
                                    model = LogisticRegression(random_state=42, max_iter=1000)
                                    model.fit(X_train_scaled, y_train)
                                else:  # SVM
                                    model = SVC(random_state=42)
                                    model.fit(X_train_scaled, y_train)
                                
                                # Predictions
                                if model_choice == "Random Forest":
                                    y_pred = model.predict(X_test)
                                else:
                                    y_pred = model.predict(X_test_scaled)
                                
                                # Results
                                accuracy = accuracy_score(y_test, y_pred)
                                st.success(f"Model Accuracy: {accuracy:.3f}")
                                
                                # Classification report
                                st.subheader("Classification Report")
                                report = classification_report(y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df)
                                
                                # Feature importance (for Random Forest)
                                if model_choice == "Random Forest":
                                    st.subheader("Feature Importance")
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_columns,
                                        'Importance': model.feature_importances_
                                    }).sort_values('Importance', ascending=False)
                                    
                                    fig = px.bar(importance_df, x='Importance', y='Feature',
                                               orientation='h', title="Feature Importance")
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.info("Regression analysis would be implemented here for continuous targets.")
                        
                except Exception as e:
                    st.error(f"Error in machine learning analysis: {e}")
        
        # PCA Analysis
        st.subheader("Principal Component Analysis")
        if len(numeric_columns) > 2:
            if st.button("Run PCA"):
                try:
                    # Standardize the data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(df[numeric_columns])
                    
                    # PCA
                    pca = PCA()
                    X_pca = pca.fit_transform(X_scaled)
                    
                    # Explained variance
                    explained_variance = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)
                    
                    # Plot explained variance
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(explained_variance) + 1)),
                        y=explained_variance,
                        mode='lines+markers',
                        name='Individual',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(cumulative_variance) + 1)),
                        y=cumulative_variance,
                        mode='lines+markers',
                        name='Cumulative',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title="PCA Explained Variance",
                        xaxis_title="Principal Component",
                        yaxis_title="Explained Variance Ratio"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # PCA scatter plot (first two components)
                    if len(numeric_columns) > 1:
                        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                                       title="PCA - First Two Components")
                        fig.update_xaxis(title="PC1")
                        fig.update_yaxis(title="PC2")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error in PCA analysis: {e}")
    
    with tab5:
        st.header("Data Table")
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            show_all = st.checkbox("Show all rows", value=False)
        with col2:
            if not show_all:
                n_rows = st.slider("Number of rows to display", 10, min(1000, len(df)), 100)
        
        # Filter options
        st.subheader("Filter Data")
        columns_to_filter = st.multiselect("Select columns to filter", df.columns.tolist())
        
        filtered_df = df.copy()
        
        for col in columns_to_filter:
            if df[col].dtype in ['int64', 'float64']:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                filter_range = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(filtered_df[col] >= filter_range[0]) & 
                                        (filtered_df[col] <= filter_range[1])]
            else:
                unique_values = df[col].unique()
                selected_values = st.multiselect(f"Filter {col}", unique_values, default=unique_values)
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
        
        # Display filtered data
        if show_all:
            st.dataframe(filtered_df)
        else:
            st.dataframe(filtered_df.head(n_rows))
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name='filtered_amazon_reviews.csv',
            mime='text/csv'
        )

else:
    st.info("Please select a data source and load your data to begin analysis.")
    
    # Information about the dataset
    st.header("📖 About the Amazon Commerce Reviews Dataset")
    st.markdown("""
    This dataset from the UCI Machine Learning Repository contains:
    
    - **Purpose**: Authorship identification in online reviews
    - **Source**: Amazon Commerce Website customer reviews
    - **Size**: 1,500 instances with 10,000 features
    - **Task**: Classification for authorship identification
    - **Features**: Linguistic style attributes including digit usage, punctuation, word and sentence length, and word frequency
    
    The dataset is designed for identifying authors of online reviews based on their writing patterns and linguistic characteristics.
    """)
    
    st.header("🚀 Getting Started")
    st.markdown("""
    1. **Sample Data**: Use the generated sample data to explore the app features
    2. **Upload CSV**: Upload your own CSV file with Amazon review data
    3. **UCI Repository**: Load the actual dataset directly from UCI (requires ucimlrepo package)
    
    Once data is loaded, you can:
    - Explore data distributions and statistics
    - Create interactive visualizations
    - Analyze feature correlations
    - Train machine learning models
    - Perform dimensionality reduction with PCA
    """)

# Footer
st.markdown("---")
st.markdown("**Dataset Citation**: Liu, Z. (2011). Amazon Commerce Reviews [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55C88")
