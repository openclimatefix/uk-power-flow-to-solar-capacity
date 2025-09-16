import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings


warnings.filterwarnings("ignore")


class YearlyCapacityValidator:

    def __init__(self, results_path: str = "scenario_results_all_locations.json"):
        self.results_path = results_path
        self.data = None
        self.monthly_data = None
        self.daily_data = None
        
    def load_results(self):
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
        
        processed_data = []
        for location, results in self.data.items():
            if location.startswith('_'):
                continue
                
            timestamps = results.get('timestamps', [])
            deltas = results.get('Delta', [])
            
            if not timestamps or not deltas:
                continue
                
            for ts, delta in zip(timestamps, deltas):
                try:
                    dt = pd.to_datetime(ts)
                    processed_data.append({
                        'location': location,
                        'timestamp': dt,
                        'solar_capacity': delta,
                        'year': dt.year,
                        'month': dt.month,
                        'day': dt.day,
                        'year_month': dt.to_period('M'),
                        'date': dt.date()
                    })
                except:
                    continue
        
        self.data_df = pd.DataFrame(processed_data)
        
    def calculate_monthly_aggregates(self):
        monthly_agg = self.data_df.groupby(['location', 'year_month']).agg({
            'solar_capacity': ['mean', 'std', 'count'],
            'year': 'first',
            'month': 'first'
        }).reset_index()
        
        monthly_agg.columns = ['location', 'year_month', 'mean_capacity', 'std_capacity', 'sample_count', 'year', 'month']
        monthly_agg = monthly_agg.sort_values(['location', 'year_month'])
        monthly_agg['capacity_change'] = monthly_agg.groupby('location')['mean_capacity'].pct_change() * 100
        monthly_agg['capacity_diff'] = monthly_agg.groupby('location')['mean_capacity'].diff()
        
        self.monthly_data = monthly_agg
        
    def calculate_daily_aggregates(self):
        daily_agg = self.data_df.groupby(['location', 'date']).agg({
            'solar_capacity': ['mean', 'count'],
            'timestamp': 'first'
        }).reset_index()
        
        daily_agg.columns = ['location', 'date', 'mean_capacity', 'sample_count', 'timestamp']
        daily_agg = daily_agg.sort_values(['location', 'date'])
        
        self.daily_data = daily_agg
        
    def calculate_yoy_mom_changes(self):
        df = self.monthly_data.copy()
        df['prev_year_capacity'] = df.groupby(['location', 'month'])['mean_capacity'].shift(1)
        df['yoy_change'] = ((df['mean_capacity'] - df['prev_year_capacity']) / df['prev_year_capacity'] * 100)
        df['mom_change'] = df.groupby('location')['mean_capacity'].pct_change() * 100
        self.monthly_data = df
        
    def save_monthly_summary(self, output_path: str = "monthly_capacity_analysis.csv"):
        if self.monthly_data is not None:
            summary = self.monthly_data.copy()
            summary['year_month_str'] = summary['year_month'].astype(str)
            summary = summary.drop('year_month', axis=1)
            summary.to_csv(output_path, index=False)
        
    def plot_time_domain_capacity(self, top_n_locations: int = 10):
        if self.data_df is None:
            return
        
        top_locations = (self.data_df.groupby('location')['solar_capacity']
                        .mean().nlargest(top_n_locations).index.tolist())
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        ax1 = axes[0]
        for location in top_locations:
            loc_data = self.data_df[self.data_df['location'] == location].sort_values('timestamp')
            ax1.plot(loc_data['timestamp'], loc_data['solar_capacity'], 
                    alpha=0.7, linewidth=1, label=location[:20] + '...' if len(location) > 20 else location)
        
        ax1.set_title('Solar Capacity Over Time - All Data Points', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Solar Capacity (MW)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        if self.daily_data is not None:
            ax2 = axes[1]
            for location in top_locations:
                loc_data = self.daily_data[self.daily_data['location'] == location].sort_values('date')
                ax2.plot(pd.to_datetime(loc_data['date']), loc_data['mean_capacity'], 
                        marker='o', markersize=2, linewidth=1.5, 
                        label=location[:20] + '...' if len(location) > 20 else location)
            
            ax2.set_title('Daily Average Solar Capacity Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Daily Mean Solar Capacity (MW)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_domain_capacity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_capacity_trends(self, top_n_locations: int = 10):
        if self.monthly_data is None:
            return
        
        top_locations = (self.monthly_data.groupby('location')['mean_capacity']
                        .mean().nlargest(top_n_locations).index.tolist())
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        ax1 = axes[0, 0]
        for location in top_locations:
            loc_data = self.monthly_data[self.monthly_data['location'] == location]
            ax1.plot(loc_data['year_month'].astype(str), loc_data['mean_capacity'], 
                    marker='o', linewidth=2, label=location[:20] + '...' if len(location) > 20 else location)
        
        ax1.set_title(f'Monthly Solar Capacity Trends - Top {top_n_locations} Locations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year-Month')
        ax1.set_ylabel('Mean Solar Capacity (MW)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        yoy_pivot = self.monthly_data.pivot_table(
            values='yoy_change', index='location', columns='year', aggfunc='mean'
        )
        yoy_pivot_top = yoy_pivot.loc[top_locations] if len(yoy_pivot) > top_n_locations else yoy_pivot
        
        sns.heatmap(yoy_pivot_top, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   center=0, ax=ax2, cbar_kws={'label': 'YoY Change (%)'})
        ax2.set_title('Year-over-Year Capacity Changes (%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Location')
        
        ax3 = axes[1, 0]
        mom_data = self.monthly_data[self.monthly_data['location'].isin(top_locations)]['mom_change'].dropna()
        ax3.boxplot(mom_data, vert=True)
        ax3.set_title('Month-over-Month Change Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('MoM Change (%)')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        seasonal_data = (self.monthly_data[self.monthly_data['location'].isin(top_locations)]
                        .groupby('month')['mean_capacity'].mean())
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax4.bar(range(1, 13), seasonal_data.values, color='skyblue', alpha=0.7)
        ax4.set_title('Seasonal Solar Capacity Patterns', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Mean Solar Capacity (MW)')
        ax4.set_xticks(range(1, 13))
        ax4.set_xticklabels(months)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('solar_capacity_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_individual_location_trends(self, locations: list = None, max_locations: int = 6):
        if self.monthly_data is None:
            return
        
        if locations is None:
            locations = (self.monthly_data.groupby('location')['mean_capacity']
                        .mean().nlargest(max_locations).index.tolist())
        else:
            locations = locations[:max_locations]
        
        n_cols = 2
        n_rows = (len(locations) + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, location in enumerate(locations):
            loc_data = self.monthly_data[self.monthly_data['location'] == location].sort_values('year_month')
            
            ax = axes[i]
            
            line1 = ax.plot(loc_data['year_month'].astype(str), loc_data['mean_capacity'], 
                           'b-', marker='o', linewidth=2, label='Solar Capacity (MW)')
            ax.set_ylabel('Solar Capacity (MW)', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax.twinx()
            line2 = ax2.plot(loc_data['year_month'].astype(str), loc_data['mom_change'], 
                            'r--', marker='s', alpha=0.7, label='MoM Change (%)')
            ax2.set_ylabel('MoM Change (%)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_title(f'{location[:30]}{"..." if len(location) > 30 else ""}', 
                        fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
        
        for j in range(len(locations), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('individual_location_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_statistics(self):
        if self.monthly_data is None:
            return
        
        print(f"Analysis Period: {self.monthly_data['year_month'].min()} to {self.monthly_data['year_month'].max()}")
        print(f"Total Locations Analyzed: {self.monthly_data['location'].nunique()}")
        print(f"Total Monthly Observations: {len(self.monthly_data)}")
        
        capacity_stats = self.monthly_data['mean_capacity'].describe()
        print(f"Mean Capacity: {capacity_stats['mean']:.3f} MW")
        print(f"Median Capacity: {capacity_stats['50%']:.3f} MW")
        print(f"Std Deviation: {capacity_stats['std']:.3f} MW")
        print(f"Range: {capacity_stats['min']:.3f} - {capacity_stats['max']:.3f} MW")
        
        mom_stats = self.monthly_data['mom_change'].describe()
        print(f"Mean MoM Change: {mom_stats['mean']:.2f}%")
        print(f"Median MoM Change: {mom_stats['50%']:.2f}%")
        print(f"MoM Change Range: {mom_stats['min']:.2f}% - {mom_stats['max']:.2f}%")
        
        yoy_stats = self.monthly_data['yoy_change'].dropna().describe()
        if not yoy_stats.empty:
            print(f"Mean YoY Change: {yoy_stats['mean']:.2f}%")
            print(f"Median YoY Change: {yoy_stats['50%']:.2f}%")
            print(f"YoY Change Range: {yoy_stats['min']:.2f}% - {yoy_stats['max']:.2f}%")
        
        print("\nTOP 10 LOCATIONS BY CAPACITY")
        top_locations = (self.monthly_data.groupby('location')['mean_capacity']
                        .mean().nlargest(10))
        for i, (location, capacity) in enumerate(top_locations.items(), 1):
            print(f"{i:2d}. {location[:40]}{'...' if len(location) > 40 else ''}: {capacity:.3f} MW")


def main():

    validator = YearlyCapacityValidator()
    
    try:
        validator.load_results()
        validator.calculate_monthly_aggregates()
        validator.calculate_daily_aggregates()
        validator.calculate_yoy_mom_changes()
        validator.save_monthly_summary()
        validator.generate_summary_statistics()
        validator.plot_time_domain_capacity(top_n_locations=10)
        validator.plot_capacity_trends(top_n_locations=10)
        validator.plot_individual_location_trends(max_locations=6)
        
        print("\nFiles created:")
        print("  - monthly_capacity_analysis.csv")
        print("  - time_domain_capacity.png")
        print("  - solar_capacity_trends.png")
        print("  - individual_location_trends.png")
        
    except FileNotFoundError:
        print(f"Results file not found: {validator.results_path}")
    except Exception as e:
        print(f"Error during validation: {str(e)}")


if __name__ == "__main__":
    main()
