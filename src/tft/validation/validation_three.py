import pandas as pd
import numpy as np
import os
import json
import requests
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings


warnings.filterwarnings('ignore')


class GSPSolarCapacityValidator:
    def __init__(self, tft_results_path: str, pvlive_data_path: str, 
                 data_directory: str, output_directory: str = "./output/"):
        self.tft_results_path = tft_results_path
        self.pvlive_data_path = pvlive_data_path
        self.data_directory = data_directory
        self.output_directory = output_directory
        
        self.gsp_df = None
        self.site_mapping = None
        self.site_solar_estimates = None
        self.gsp_system_df = None
        self.pvlive_gsp_summary = None
        self.analysis_df = None
        
        os.makedirs(output_directory, exist_ok=True)
        
    def load_gsp_data(self):
        url = "https://raw.githubusercontent.com/openclimatefix/ocf-data-sampler/main/ocf_data_sampler/data/uk_gsp_locations_20250109.csv"
        try:
            response = requests.get(url)
            response.raise_for_status()
            self.gsp_df = pd.read_csv(StringIO(response.text))
        except Exception as e:
            os.system("git clone --depth 1 https://github.com/openclimatefix/ocf-data-sampler.git")
            self.gsp_df = pd.read_csv("ocf-data-sampler/ocf_data_sampler/data/uk_gsp_locations_20250109.csv")
    
    def load_site_data(self):
        all_sites_hourly_data = {}
        parquet_files = [os.path.join(self.data_directory, f) 
                        for f in os.listdir(self.data_directory) if f.endswith('.parquet')]
        
        for file_path in parquet_files:
            temp_df = pd.read_parquet(file_path)
            for site_name, site_df in temp_df.groupby('location'):
                site_df = site_df.set_index('timestamp')
                site_df.index = pd.to_datetime(site_df.index)
                all_sites_hourly_data[site_name] = {
                    'power': site_df['active_power_mw'],
                    'coords': {'latitude': site_df['latitude'].iloc[0], 
                              'longitude': site_df['longitude'].iloc[0]}
                }
            del temp_df
        
        return all_sites_hourly_data
    
    def map_sites_to_gsp(self, sites_data):
        gsp_gdf = gpd.GeoDataFrame(self.gsp_df, 
                                  geometry=gpd.GeoSeries.from_wkt(self.gsp_df['geometry']), 
                                  crs='EPSG:27700')
        gsp_gdf_regional = gsp_gdf[gsp_gdf['gsp_id'] != 0].copy()
        
        site_gsp_mapping = {}
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        
        for site_name, site_info in sites_data.items():
            lat, lon = site_info['coords']['latitude'], site_info['coords']['longitude']
            x, y = transformer.transform(lon, lat)
            site_point = Point(x, y)
            containing_gsp_series = gsp_gdf_regional[gsp_gdf_regional.geometry.contains(site_point)]
            
            if not containing_gsp_series.empty:
                gsp_info_row = containing_gsp_series.iloc[0]
                site_gsp_mapping[site_name] = {
                    'gsp_id': gsp_info_row['gsp_id'], 
                    'gsp_name': gsp_info_row['gsp_name']
                }
        
        return site_gsp_mapping
    
    def load_tft_results(self):
        with open(self.tft_results_path, 'r') as f:
            tft_results = json.load(f)
        
        site_solar_estimates = {}
        for site_name, data in tft_results.items():
            if isinstance(data, dict) and 'Predicted_Delta' in data:
                max_capacity = max(np.nanmax(data.get('Predicted_Delta', [0])), 0)
                site_solar_estimates[site_name] = {
                    'max_solar_capacity_mw': max_capacity,
                    'mean_predicted_delta': data.get('mean_predicted_delta', 0),
                    'proxy_used': data.get('proxy_used', 'unknown')
                }
        
        return site_solar_estimates
    
    def aggregate_by_gsp(self):
        gsp_solar_aggregated = {}
        for site_name, site_info in self.site_mapping.items():
            gsp_id, gsp_name = site_info['gsp_id'], site_info['gsp_name']
            if gsp_id not in gsp_solar_aggregated:
                gsp_solar_aggregated[gsp_id] = {
                    'gsp_name': gsp_name, 
                    'site_capacities': [], 
                    'site_details': []
                }
            
            if site_name in self.site_solar_estimates:
                capacity = self.site_solar_estimates[site_name]['max_solar_capacity_mw']
                proxy = self.site_solar_estimates[site_name]['proxy_used']
                
                gsp_solar_aggregated[gsp_id]['site_capacities'].append(capacity)
                gsp_solar_aggregated[gsp_id]['site_details'].append({
                    'site': site_name,
                    'capacity': capacity,
                    'proxy': proxy
                })
        
        gsp_summary_data = []
        for gsp_id, data in gsp_solar_aggregated.items():
            if data['site_capacities']:
                caps = np.array(data['site_capacities'])
                gsp_summary_data.append({
                    'gsp_id': gsp_id, 
                    'gsp_name': data['gsp_name'],
                    'sites_in_sample': len(caps), 
                    'sample_capacity_mw': caps.sum(),
                    'mean_site_capacity': caps.mean(),
                    'max_site_capacity': caps.max(),
                    'min_site_capacity': caps.min()
                })
        
        return pd.DataFrame(gsp_summary_data)
    
    def fetch_official_capacity(self):
        ACCESS_TOKEN = "PLACEHOLDER"
        headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
        
        try:
            system_url = "https://api.quartz.solar/v0/system/GB/gsp/"
            response = requests.get(system_url, headers=headers, timeout=30)
            if response.status_code == 200:
                return pd.DataFrame(response.json())
            else:
                return None
        except Exception as e:
            return None
    
    def process_pvlive_data(self):
        try:
            pvlive_df = pd.read_csv(self.pvlive_data_path)
            pvlive_filtered = pvlive_df[pvlive_df['solar_generation_mw'] > 0]
            return pvlive_filtered.groupby('gsp_id').agg(
                Peak_MW=('solar_generation_mw', 'max')
            ).reset_index()
        except Exception as e:
            return None
    
    def create_validation_analysis(self, gsp_sample_df):
        if self.gsp_system_df is None or self.pvlive_gsp_summary is None:
            return None
        
        analysis_df = gsp_sample_df.merge(
            self.gsp_system_df[['gspId', 'installedCapacityMw']], 
            left_on='gsp_id', right_on='gspId', how='inner'
        )
        analysis_df = analysis_df.merge(self.pvlive_gsp_summary, on='gsp_id', how='inner')
        analysis_df.dropna(subset=['installedCapacityMw', 'sample_capacity_mw', 'Peak_MW'], inplace=True)
        
        analysis_df['scaled_estimate_mw'] = analysis_df['installedCapacityMw']
        analysis_df['capacity_vs_peak_ratio'] = analysis_df['scaled_estimate_mw'] / analysis_df['Peak_MW']
        analysis_df['sample_vs_official_ratio'] = analysis_df['sample_capacity_mw'] / analysis_df['installedCapacityMw']
        
        return analysis_df
    
    def calculate_validation_metrics(self):
        if self.analysis_df is None:
            return {}
        
        total_sample_capacity = self.analysis_df['sample_capacity_mw'].sum()
        total_installed_capacity = self.analysis_df['installedCapacityMw'].sum()
        total_peak_generation = self.analysis_df['Peak_MW'].sum()
        total_sites_analyzed = self.analysis_df['sites_in_sample'].sum()
        
        scaling_factor = total_installed_capacity / total_sample_capacity if total_sample_capacity > 0 else 0
        final_ratio = total_installed_capacity / total_peak_generation if total_peak_generation > 0 else 0
        correlation = self.analysis_df[['installedCapacityMw', 'Peak_MW']].corr().iloc[0, 1]
        
        return {
            'total_sites_analyzed': total_sites_analyzed,
            'total_sample_capacity': total_sample_capacity,
            'total_installed_capacity': total_installed_capacity,
            'total_peak_generation': total_peak_generation,
            'scaling_factor': scaling_factor,
            'final_ratio': final_ratio,
            'correlation': correlation,
            'gsp_regions_with_data': len(self.analysis_df)
        }
    
    def create_validation_plots(self, metrics):
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle('GSP Solar Capacity: Validation Analysis', fontsize=16, fontweight='bold')
        
        top_15 = self.analysis_df.sort_values('installedCapacityMw', ascending=False).head(15)
        
        x_pos = np.arange(len(top_15))
        axes[0, 0].bar(x_pos - 0.2, top_15['sample_capacity_mw'], 0.4, 
                      label='Sample Estimate', color='lightcoral', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.2, top_15['installedCapacityMw'], 0.4, 
                      label='Official Installed Capacity', color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Sample vs Official Capacity (Top 15 GSPs)')
        axes[0, 0].set_ylabel('Capacity (MW)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([name[:8] for name in top_15['gsp_name']], 
                                  rotation=45, ha='right')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, which="both", ls="--", alpha=0.5)
        
        axes[0, 1].scatter(self.analysis_df['Peak_MW'], self.analysis_df['installedCapacityMw'], 
                          alpha=0.7, c='green', edgecolors='k', s=60)
        max_val = max(self.analysis_df['Peak_MW'].max(), 
                     self.analysis_df['installedCapacityMw'].max()) * 1.05
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')
        axes[0, 1].set_title(f'Validation: Capacity vs Generation (Corr: {metrics["correlation"]:.3f})')
        axes[0, 1].set_xlabel('Actual Peak Generation (MW)')
        axes[0, 1].set_ylabel('Official Installed Capacity (MW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].hist(self.analysis_df['sites_in_sample'], bins=20, 
                       color='orange', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Distribution of Sites per GSP')
        axes[0, 2].set_xlabel('Number of Sites')
        axes[0, 2].set_ylabel('Number of GSPs')
        axes[0, 2].grid(True, alpha=0.3)
        
        mean_sites = self.analysis_df['sites_in_sample'].mean()
        axes[0, 2].axvline(mean_sites, color='red', linestyle='--', 
                          label=f'Mean: {mean_sites:.1f}')
        axes[0, 2].legend()
        
        mean_ratio_val = self.analysis_df['capacity_vs_peak_ratio'].mean()
        axes[1, 0].hist(self.analysis_df['capacity_vs_peak_ratio'].dropna(), bins=20, 
                       color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(mean_ratio_val, color='red', linestyle='--', 
                          label=f'Mean Ratio: {mean_ratio_val:.2f}')
        axes[1, 0].set_title('Distribution of (Capacity / Peak Generation) Ratio')
        axes[1, 0].set_xlabel('Ratio')
        axes[1, 0].set_ylabel('Number of GSPs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(self.analysis_df['sites_in_sample'], 
                          self.analysis_df['sample_capacity_mw'], 
                          alpha=0.7, c='brown', edgecolors='k', s=60)
        axes[1, 1].set_title('Sample Capacity vs Number of Sites')
        axes[1, 1].set_xlabel('Number of Sites in GSP')
        axes[1, 1].set_ylabel('Sample Capacity (MW)')
        axes[1, 1].grid(True, alpha=0.3)
        
        site_capacity_corr = self.analysis_df[['sites_in_sample', 'sample_capacity_mw']].corr().iloc[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {site_capacity_corr:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle="round", facecolor='wheat'))
        
        axes[1, 2].axis('off')
        summary_text = (
                       f"Total Sites Analyzed: {metrics['total_sites_analyzed']}\n"
                       f"GSP Regions with Data: {metrics['gsp_regions_with_data']}\n"
                       f"Average Sites per GSP: {self.analysis_df['sites_in_sample'].mean():.1f}\n\n"
                       f"Total Sample Capacity: {metrics['total_sample_capacity']:.1f} MW\n"
                       f"Total Official Capacity: {metrics['total_installed_capacity']:.1f} MW\n"
                       f"Total Peak Generation: {metrics['total_peak_generation']:.1f} MW\n\n"
                       f"--- KEY METRICS ---\n"
                       f"Scaling Factor: {metrics['scaling_factor']:.0f}x\n"
                       f"Capacity/Generation Ratio: {metrics['final_ratio']:.2f}\n"
                       f"Validation Correlation: {metrics['correlation']:.3f}\n\n"
                       f"Mean Capacity per Site: {metrics['total_sample_capacity']/metrics['total_sites_analyzed']:.3f} MW"
                       )
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace', 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = os.path.join(self.output_directory, 'gsp_validation_analysis.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def analyze_proxy_performance(self):
        proxy_analysis = {}
        for site_name, estimates in self.site_solar_estimates.items():
            proxy = estimates['proxy_used']
            capacity = estimates['max_solar_capacity_mw']
            
            if proxy not in proxy_analysis:
                proxy_analysis[proxy] = []
            proxy_analysis[proxy].append(capacity)
        
        proxy_summary = []
        for proxy, capacities in proxy_analysis.items():
            proxy_summary.append({
                'proxy_location': proxy,
                'sites_using_proxy': len(capacities),
                'mean_capacity': np.mean(capacities),
                'total_capacity': np.sum(capacities),
                'std_capacity': np.std(capacities)
            })
        
        return pd.DataFrame(proxy_summary).sort_values('total_capacity', ascending=False)
    
    def save_results(self, gsp_sample_df, proxy_df, metrics):
        site_estimates_df = pd.DataFrame.from_dict(self.site_solar_estimates, orient='index')
        site_estimates_path = os.path.join(self.output_directory, 'site_solar_estimates.csv')
        site_estimates_df.to_csv(site_estimates_path, index_label='site_name')
        
        if self.analysis_df is not None:
            analysis_path = os.path.join(self.output_directory, 'gsp_validation_analysis.csv')
            self.analysis_df.to_csv(analysis_path, index=False)
        
        proxy_path = os.path.join(self.output_directory, 'proxy_performance_analysis.csv')
        proxy_df.to_csv(proxy_path, index=False)
        
        metrics_path = os.path.join(self.output_directory, 'validation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def run_full_validation(self):
        self.load_gsp_data()
        all_sites_data = self.load_site_data()
        self.site_mapping = self.map_sites_to_gsp(all_sites_data)
        self.site_solar_estimates = self.load_tft_results()
        gsp_sample_df = self.aggregate_by_gsp()
        self.gsp_system_df = self.fetch_official_capacity()
        self.pvlive_gsp_summary = self.process_pvlive_data()
        self.analysis_df = self.create_validation_analysis(gsp_sample_df)
        metrics = self.calculate_validation_metrics()
        
        if self.analysis_df is not None:
            plot_path = self.create_validation_plots(metrics)
        
        proxy_df = self.analyze_proxy_performance()
        self.save_results(gsp_sample_df, proxy_df, metrics)
        
        return {
            'gsp_analysis': self.analysis_df,
            'proxy_analysis': proxy_df,
            'metrics': metrics,
            'site_mapping_count': len(self.site_mapping),
            'site_estimates_count': len(self.site_solar_estimates)
        }

def main():

    validator = GSPSolarCapacityValidator(
        tft_results_path="./output/tft_all_sites_scenario_results.json",
        pvlive_data_path="/home/felix/pvlive_parsed_data.csv",
        data_directory="/home/felix/post-fe-data/",
        output_directory="./output/"
    )
    
    results = validator.run_full_validation()
    print(f"Sites with GSP mapping: {results['site_mapping_count']}")
    print(f"Sites with solar estimates: {results['site_estimates_count']}")
    
    if results['metrics']:
        metrics = results['metrics']
        print(f"Total sites analyzed: {metrics['total_sites_analyzed']}")
        print(f"GSP regions: {metrics['gsp_regions_with_data']}")
        print(f"Validation correlation: {metrics['correlation']:.3f}")
        print(f"Capacity/Generation ratio: {metrics['final_ratio']:.2f}")


if __name__ == "__main__":
    main()
