from medAI.datasets.optimum.utils import get_cleaned_ua_metadata_table
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd


class OptimumEvaluator: 
    def __init__(self, ai_model_outputs): 
        columns = ['average_needle_heatmap_value', 'image_level_cancer_logits', 'involvement']
        ai_model_outputs = ai_model_outputs.set_index('core_id')[columns]
        self.ai_model_outputs = ai_model_outputs

    def get_table_with_ai_outputs(self, *args, **kwargs):
        """
        Returns a table with AI model outputs joined to the cleaned UA metadata table.
        """
        table = get_cleaned_ua_metadata_table(*args, **kwargs)
        table = table.join(self.ai_model_outputs, how='left')
        table = table.loc[~table['image_level_cancer_logits'].isna()]

        import numpy as np

        model_cspca_score = table['image_level_cancer_logits']
        counts = table['PRI-MUS'].value_counts(normalize=True)

        bins = [0]
        cum = 0
        for i in range(1, 6):
            prob = counts.at[i]
            cum += prob
            bins.append(np.quantile(model_cspca_score, cum))

        table['binned_model_cspca_score'] = pd.cut(model_cspca_score, bins=bins, labels=range(1, 6)).astype(float)
        table = table.loc[~table['binned_model_cspca_score'].isna()]
        #pd.cut(model_cspca_score, bins=bins).value_counts(normalize=True)

        return table

    def compute_aurocs(self, gg_greater_than=3):
        tab = self.get_table_with_ai_outputs(filter_has_primus=True, filter_has_pirads=False)
        n = len(tab)

        t = gg_greater_than

        metrics = {}
        metrics['PRI-MUS_auroc'] = roc_auc_score(
            tab['grade_group'] >= t, tab['PRI-MUS']
        )
        metrics['hmap_auroc'] = roc_auc_score(
            tab['grade_group'] >= t, tab['average_needle_heatmap_value']
        )
        metrics['model_auroc'] = roc_auc_score(
            tab['grade_group'] >= t, tab['image_level_cancer_logits']
        )
        metrics['n'] = n

        return metrics

