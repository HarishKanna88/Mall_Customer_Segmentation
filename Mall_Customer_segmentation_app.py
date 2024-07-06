import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r"C:\Users\91979\desktop\.streamlit\Mall_Customers.csv")

# Function to preprocess data
def preprocess_data(data):
    # Drop any missing values
    data.dropna(inplace=True)
    
    # Encode categorical variables if necessary (e.g., Gender)
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    
    # Feature Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
    data_scaled = pd.DataFrame(data_scaled, columns=['Age', 'Annual Income', 'Spending Score'])
    
    return data_scaled, scaler

data_scaled, scaler = preprocess_data(data)

# Function to apply K-Means clustering
def apply_kmeans(data_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    return data, kmeans

data, kmeans = apply_kmeans(data_scaled)

# Define custom color palette for clusters
cluster_palette = sns.color_palette('Set1', n_colors=len(data['Cluster'].unique()))

# Function to create and show scatter plot with colors
def show_scatter_plot(selected_feature):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=selected_feature, y='Spending Score (1-100)', hue='Cluster', data=data, palette=cluster_palette)
    plt.title('Clusters of Customers')
    plt.xlabel(selected_feature)
    plt.ylabel('Spending Score (1-100)')
    st.pyplot(plt)

# Function to create and show age distribution plot
def show_age_distribution(selected_feature):
    plt.figure(figsize=(10, 8))
    for cluster in sorted(data['Cluster'].unique()):  # Sort clusters numerically
        cluster_data = data[data['Cluster'] == cluster]
        sns.histplot(cluster_data[selected_feature], kde=True, label=f'Cluster {cluster}', color=cluster_palette[cluster])
    plt.title(f'{selected_feature} Distribution in Each Cluster')
    plt.legend()
    st.pyplot(plt)

# Function to show descriptive statistics
def show_descriptive_stats():
    st.subheader('Descriptive Statistics')
    st.write(data.describe())

# Function to show insights based on clusters in numerical order
def show_insights():
    st.subheader('Insights for Each Cluster')
    for cluster in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster]
        st.write(f'**Cluster {cluster}:**')
        st.write(f'- Number of Customers: {len(cluster_data)}')
        st.write(f'- Mean Age: {cluster_data["Age"].mean():.2f}')
        st.write(f'- Mean Annual Income: {cluster_data["Annual Income (k$)"].mean():.2f}')
        st.write(f'- Mean Spending Score: {cluster_data["Spending Score (1-100)"].mean():.2f}')
        st.write('---')

# Streamlit app
def main():
    st.title('Customer Segmentation Analysis')

    # Sidebar for data exploration options
    st.sidebar.title('Data Exploration Options')
    selected_feature = st.sidebar.selectbox('Select Feature', ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    # Show scatter plot based on selected feature
    st.header('Clusters of Customers')
    show_scatter_plot(selected_feature)

    # Show age distribution based on selected feature
    st.header(f'{selected_feature} Distribution in Each Cluster')
    show_age_distribution(selected_feature)

    # Show descriptive statistics
    show_descriptive_stats()

    # Show insights for each cluster
    show_insights()

if __name__ == "__main__":
    main()
