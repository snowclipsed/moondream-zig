import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

def parse_profile_log(file_path):
    """Parse the profiling log file into a pandas DataFrame."""
    operations = []
    times = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if '[VISION PROFILE]' in line:
                # Extract operation name and time using regex
                match = re.search(r'\[VISION PROFILE\] (.*?): (\d+\.\d+)ms', line)
                if match:
                    operation = match.group(1)
                    time = float(match.group(2))
                    operations.append(operation)
                    times.append(time)
    
    return pd.DataFrame({
        'Operation': operations,
        'Time (ms)': times
    })

def create_bar_chart(df, output_dir):
    """Create a horizontal bar chart of operation times."""
    plt.figure(figsize=(12, max(8, len(df) * 0.4)))
    
    # Sort by time in descending order
    df_sorted = df.sort_values('Time (ms)', ascending=True)
    
    # Create horizontal bar chart
    bars = plt.barh(df_sorted['Operation'], df_sorted['Time (ms)'])
    plt.title('Operation Times (ms)', pad=20)
    plt.xlabel('Time (milliseconds)')
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}ms', 
                va='center', ha='left', fontsize=8)
    
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'operation_times_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pie_chart(df, output_dir):
    """Create a pie chart showing proportion of time spent in each operation."""
    plt.figure(figsize=(12, 8))
    
    # Filter out operations that take less than 1% of total time for readability
    total_time = df['Time (ms)'].sum()
    significant_ops = df[df['Time (ms)'] / total_time >= 0.01]
    other_time = df[df['Time (ms)'] / total_time < 0.01]['Time (ms)'].sum()
    
    if other_time > 0:
        significant_ops = pd.concat([
            significant_ops,
            pd.DataFrame({
                'Operation': ['Other'],
                'Time (ms)': [other_time]
            })
        ])
    
    plt.pie(significant_ops['Time (ms)'], labels=significant_ops['Operation'], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Proportion of Time Spent in Each Operation')
    plt.axis('equal')
    plt.savefig(output_dir / 'operation_times_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cumulative_chart(df, output_dir):
    """Create a cumulative time chart."""
    plt.figure(figsize=(12, 6))
    
    # Sort by time and calculate cumulative sum
    df_sorted = df.sort_values('Time (ms)', ascending=True)
    df_sorted['Cumulative Time (ms)'] = df_sorted['Time (ms)'].cumsum()
    
    plt.plot(range(len(df_sorted)), df_sorted['Cumulative Time (ms)'], 
             marker='o', linestyle='-')
    
    # Rotate labels for better readability
    plt.xticks(range(len(df_sorted)), df_sorted['Operation'], 
               rotation=45, ha='right')
    
    plt.title('Cumulative Operation Time')
    plt.ylabel('Cumulative Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(df_sorted['Cumulative Time (ms)']):
        plt.text(i, v, f'{v:.2f}ms', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_time.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistics_report(df, output_dir):
    """Generate a text report with statistical analysis."""
    total_time = df['Time (ms)'].sum()
    avg_time = df['Time (ms)'].mean()
    median_time = df['Time (ms)'].median()
    max_op = df.loc[df['Time (ms)'].idxmax()]
    min_op = df.loc[df['Time (ms)'].idxmin()]
    
    report = f"""Profile Analysis Report
======================
Total Time: {total_time:.2f}ms
Average Operation Time: {avg_time:.2f}ms
Median Operation Time: {median_time:.2f}ms

Slowest Operation: {max_op['Operation']} ({max_op['Time (ms)']:.2f}ms)
Fastest Operation: {min_op['Operation']} ({min_op['Time (ms)']:.2f}ms)

Operations by Time (Descending):
"""
    
    for _, row in df.sort_values('Time (ms)', ascending=False).iterrows():
        report += f"- {row['Operation']}: {row['Time (ms)']:.2f}ms ({row['Time (ms)']/total_time*100:.1f}%)\n"
    
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report)

def main():
    # Create output directory
    output_dir = Path('profile_analysis')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Read and parse the profile log
        df = parse_profile_log('profile1.txt')
        
        # Generate visualizations
        create_bar_chart(df, output_dir)
        create_pie_chart(df, output_dir)
        create_cumulative_chart(df, output_dir)
        generate_statistics_report(df, output_dir)
        
        print(f"Analysis complete. Results saved in {output_dir}/")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()