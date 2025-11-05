"""
Biological Predictability Analysis for H&E to Orion Marker Prediction

This script analyzes which Orion markers are likely predictable from H&E morphology
based on known biological relationships and data-driven analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Load the Orion marker panel
def load_marker_info():
    """Load and categorize Orion markers by biological function"""
    marker_df = pd.read_csv('data/paired_core_extracted_TA118/orion_macrophage_pannel.csv')
    
    # Biological predictability categories (based on H&E morphology relationship)
    predictability_map = {
        # High predictability - structural/morphological markers
        'IBA1': 'high',      # Pan-macrophage, should correlate with cell morphology
        'CD163': 'high',     # Broad macrophage, morphologically distinct
        'HLA-DR': 'medium',  # Dendritic cells, some morphological features
        'CD3e': 'high',      # T cells, lymphocyte morphology
        'CD8a': 'medium',    # CD8+ T cells, subset of lymphocytes
        'Pan-CK': 'high',    # Epithelial/cancer cells, distinct morphology
        'SMA': 'high',       # Smooth muscle, structural features
        
        # Medium predictability - some morphological correlation
        'SPP1': 'medium',    # TAM subset, may correlate with location/morphology
        'FOLR2': 'low',      # Functional marker, less morphological correlation
        'NLRP3': 'low',      # Inflammasome, functional rather than morphological
        'LYVE1': 'medium',   # Tissue-resident subset, may have spatial patterns
        'IL-4I1': 'low',     # Functional/immunoregulatory marker
        
        # Low predictability - functional markers
        'FAP': 'medium',     # Fibroblast marker, some structural correlation
        'GFPT2': 'low',      # Metabolic marker
        'FOXP3': 'low',      # Transcription factor, functional
        'CD15': 'medium',    # Neutrophil marker, some morphological features
    }
    
    marker_df['predicted_difficulty'] = marker_df['protein'].map(predictability_map)
    return marker_df

def empirical_predictability_analysis(he_features, orion_expressions):
    """
    Empirical analysis of marker predictability using simple models
    
    Args:
        he_features: Extracted H&E image features (morphology, texture, etc.)
        orion_expressions: Ground truth Orion marker expressions
    
    Returns:
        Dictionary with predictability scores per marker
    """
    results = {}
    
    for marker in orion_expressions.columns:
        if marker == 'cell_id':  # Skip ID columns
            continue
            
        # Simple Random Forest to estimate upper bound
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Cross-validation to avoid overfitting
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(rf, he_features, orion_expressions[marker], 
                                   cv=5, scoring='r2')
        
        # Correlation analysis
        rf.fit(he_features, orion_expressions[marker])
        pred = rf.predict(he_features)
        correlation, p_value = spearmanr(pred, orion_expressions[marker])
        
        results[marker] = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'correlation': correlation,
            'p_value': p_value,
            'feature_importance': rf.feature_importances_,
            'predictability_category': categorize_predictability(cv_scores.mean())
        }
    
    return results

def categorize_predictability(r2_score):
    """Categorize predictability based on R2 score"""
    if r2_score > 0.5:
        return 'high'
    elif r2_score > 0.25:
        return 'medium'
    else:
        return 'low'

def visualize_predictability(results):
    """Create visualization of marker predictability"""
    markers = list(results.keys())
    r2_scores = [results[m]['cv_r2_mean'] for m in markers]
    correlations = [results[m]['correlation'] for m in markers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R2 scores
    colors = ['green' if r > 0.5 else 'orange' if r > 0.25 else 'red' for r in r2_scores]
    ax1.barh(markers, r2_scores, color=colors)
    ax1.set_xlabel('Cross-Validation R² Score')
    ax1.set_title('Marker Predictability from H&E Features')
    ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.7, label='High')
    ax1.axvline(x=0.25, color='orange', linestyle='--', alpha=0.7, label='Medium')
    ax1.legend()
    
    # Correlation vs R2
    ax2.scatter(r2_scores, correlations, c=colors, s=100, alpha=0.7)
    for i, marker in enumerate(markers):
        ax2.annotate(marker, (r2_scores[i], correlations[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('R² Score')
    ax2.set_ylabel('Spearman Correlation')
    ax2.set_title('Predictability Analysis')
    
    plt.tight_layout()
    plt.savefig('marker_predictability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_project_recommendations(results):
    """Generate specific recommendations for project goals"""
    
    high_pred = [m for m, r in results.items() if r['predictability_category'] == 'high']
    medium_pred = [m for m, r in results.items() if r['predictability_category'] == 'medium']
    low_pred = [m for m, r in results.items() if r['predictability_category'] == 'low']
    
    recommendations = {
        'focus_markers': high_pred,  # Focus deep learning efforts here
        'secondary_markers': medium_pred,  # Include but don't expect perfect results
        'research_markers': low_pred,  # Interesting for novel discovery but challenging
        'model_strategy': {
            'primary_loss_weight': {marker: 1.0 for marker in high_pred},
            'secondary_loss_weight': {marker: 0.5 for marker in medium_pred},
            'research_loss_weight': {marker: 0.1 for marker in low_pred}
        },
        'evaluation_strategy': {
            'success_threshold_high': 0.6,  # R² for high predictability markers
            'success_threshold_medium': 0.4,  # R² for medium predictability markers
            'success_threshold_low': 0.2,   # R² for low predictability markers
        }
    }
    
    return recommendations

if __name__ == "__main__":
    # Load marker information
    marker_info = load_marker_info()
    print("Marker Biological Categories:")
    print(marker_info.groupby('predicted_difficulty').size())
    
    # TODO: Load actual H&E features and Orion expressions for empirical analysis
    # he_features, orion_expressions = load_paired_data()
    # results = empirical_predictability_analysis(he_features, orion_expressions)
    # visualize_predictability(results)
    # recommendations = generate_project_recommendations(results)
    
    print("\n=== PROJECT STRATEGY RECOMMENDATIONS ===")
    print("Focus on these high-predictability markers first:")
    print("- IBA1 (pan-macrophage, morphologically distinct)")
    print("- CD163 (broad macrophage)")
    print("- CD3e (T cell morphology)")
    print("- Pan-CK (epithelial/cancer cell morphology)")
    print("- SMA (structural features)")
    
    print("\nSecondary targets:")
    print("- SPP1, LYVE1 (spatial patterns)")
    print("- HLA-DR, CD8a (subset morphology)")
    
    print("\nResearch targets (expect lower performance):")
    print("- FOLR2, NLRP3, IL-4I1 (functional markers)")
    print("- FOXP3, GFPT2 (transcriptional/metabolic)") 