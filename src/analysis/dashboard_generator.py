#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hockey Prediction System - Master Dashboard Generator
====================================================

Generates consolidated executive dashboard from all specialized notebooks.
Combines insights from main_analysis, strategy_optimization, risk_assessment, model_validation.

Author: Hockey Prediction System
Location: src/analysis/dashboard_generator.py  
Used by: run_all_notebooks.py
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import html


class MasterDashboardGenerator:
    """
    Generates consolidated executive dashboard from individual notebook summaries.
    
    Combines outputs from:
    - main_analysis.ipynb
    - strategy_optimization.ipynb  
    - risk_assessment.ipynb
    - model_validation.ipynb
    """
    
    def __init__(self, run_dir: Path, run_timestamp: str):
        """
        Initialize dashboard generator.
        
        Args:
            run_dir: Directory containing notebook outputs
            run_timestamp: Timestamp for this analysis run
        """
        self.run_dir = Path(run_dir)
        self.run_timestamp = run_timestamp
        self.summaries_dir = self.run_dir / 'individual_summaries'
        self.charts_dir = self.run_dir / 'all_charts'
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize summary data containers
        self.summaries = {}
        self.consolidated_data = {
            'overview': {},
            'performance': {},
            'optimization': {},
            'risk_assessment': {},
            'model_validation': {},
            'recommendations': {},
            'charts': []
        }
        
    def load_individual_summaries(self) -> bool:
        """
        Load JSON summaries from all executed notebooks with improved error handling.
        
        Returns:
            bool: True if at least one summary was loaded
        """
        summary_files = [
            'main_analysis',
            'strategy_optimization', 
            'risk_assessment',
            'model_validation'
        ]
        
        loaded_count = 0
        
        for summary_name in summary_files:
            # Look for files with various patterns (summary, executive, etc.)
            patterns = [
                f"{summary_name}_*summary*.json",
                f"{summary_name}_*executive*.json", 
                f"{summary_name}_*.json"
            ]
            
            matching_files = []
            for pattern in patterns:
                matching_files.extend(list(self.summaries_dir.glob(pattern)))
            
            if matching_files:
                # Use most recent file if multiple exist
                summary_file = max(matching_files, key=lambda p: p.stat().st_mtime)
                
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.summaries[summary_name] = data
                    
                    self.logger.info(f"✅ Loaded summary: {summary_file.name}")
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {summary_file}: {e}")
                    try:
                        # Try to read as text and show first few lines for debugging
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            content = f.read()[:200]
                            self.logger.warning(f"📄 File preview: {content}...")
                    except:
                        pass
            else:
                self.logger.warning(f"⚠️ Summary not found: {summary_name}")
                # Create placeholder data to prevent crashes
                self.summaries[summary_name] = {
                    'notebook_status': 'failed',
                    'analysis_period': 'N/A',
                    'total_games': 0,
                    'overall_roi': 0.0
                }
                
        self.logger.info(f"📄 Loaded {loaded_count}/{len(summary_files)} summaries")
        return loaded_count > 0 or len(self.summaries) > 0
        
    def consolidate_overview_data(self):
        """Consolidate high-level overview information."""
        overview = self.consolidated_data['overview']
        
        # From main_analysis summary
        if 'main_analysis' in self.summaries:
            main_data = self.summaries['main_analysis']
            overview.update({
                'analysis_period': main_data.get('analysis_period', 'N/A'),
                'total_games': main_data.get('total_games', 0),
                'total_predictions': main_data.get('total_predictions', 0),
                'data_quality_score': main_data.get('data_quality_score', 0.0)
            })
            
        # Add timestamp and run info
        overview.update({
            'run_timestamp': self.run_timestamp,
            'generated_at': datetime.now().isoformat(),
            'notebooks_executed': len([s for s in self.summaries.values() if s.get('notebook_status') != 'failed'])
        })
        
    def consolidate_performance_data(self):
        """Consolidate performance metrics from all notebooks."""
        performance = self.consolidated_data['performance']
        
        # Main analysis performance
        if 'main_analysis' in self.summaries:
            main_data = self.summaries['main_analysis']
            performance.update({
                'overall_roi': main_data.get('overall_roi', 0.0),
                'win_rate': main_data.get('win_rate', 0.0),
                'total_profit': main_data.get('total_profit', 0.0),
                'number_of_bets': main_data.get('number_of_bets', 0)
            })
            
        # Model validation performance  
        if 'model_validation' in self.summaries:
            validation_data = self.summaries['model_validation']
            performance.update({
                'model_accuracy': validation_data.get('overall_accuracy', 0.0),
                'calibration_score': validation_data.get('calibration_score', 0.0),
                'brier_score': validation_data.get('brier_score', 0.0)
            })
            
    def consolidate_optimization_data(self):
        """Consolidate strategy optimization results."""
        optimization = self.consolidated_data['optimization']
        
        if 'strategy_optimization' in self.summaries:
            opt_data = self.summaries['strategy_optimization']
            optimization.update({
                'best_strategy': opt_data.get('best_strategy', {}),
                'optimal_parameters': opt_data.get('optimal_parameters', {}),
                'improvement_potential': opt_data.get('improvement_potential', 0.0),
                'parameter_sensitivity': opt_data.get('parameter_sensitivity', {})
            })
            
    def consolidate_risk_data(self):
        """Consolidate risk assessment results."""
        risk = self.consolidated_data['risk_assessment']
        
        if 'risk_assessment' in self.summaries:
            risk_data = self.summaries['risk_assessment']
            # Handle different possible keys in risk assessment data
            risk.update({
                'max_drawdown': risk_data.get('max_drawdown', risk_data.get('worst_drawdown', 0.0)),
                'var_95': risk_data.get('var_95', risk_data.get('average_var_95', 0.0)),
                'sharpe_ratio': risk_data.get('sharpe_ratio', risk_data.get('average_sharpe_ratio', 0.0)),
                'risk_score': risk_data.get('risk_score', risk_data.get('overall_risk_level', 0.0)),
                'stress_test_results': risk_data.get('stress_test_results', {}),
                'portfolio_score': risk_data.get('portfolio_score', 0.0),
                'decision': risk_data.get('decision', 'Unknown')
            })
            
    def consolidate_validation_data(self):
        """Consolidate model validation results."""
        validation = self.consolidated_data['model_validation']
        
        if 'model_validation' in self.summaries:
            val_data = self.summaries['model_validation']
            validation.update({
                'deployment_readiness': val_data.get('deployment_readiness', 'Unknown'),
                'accuracy_trend': val_data.get('accuracy_trend', 'Stable'),
                'calibration_quality': val_data.get('calibration_quality', 'Good'),
                'market_comparison': val_data.get('market_comparison', {})
            })
            
    def generate_recommendations(self):
        """Generate consolidated business recommendations."""
        recommendations = []
        
        # Performance-based recommendations
        performance = self.consolidated_data['performance']
        roi = performance.get('overall_roi', 0.0)
        
        if roi > 0.10:  # >10% ROI
            recommendations.append({
                'type': 'positive',
                'category': 'Performance', 
                'message': f'Excellent ROI of {roi:.1%} achieved. Consider scaling investment.',
                'priority': 'high'
            })
        elif roi > 0.05:  # 5-10% ROI
            recommendations.append({
                'type': 'neutral',
                'category': 'Performance',
                'message': f'Good ROI of {roi:.1%}. Monitor performance and optimize further.',
                'priority': 'medium'
            })
        else:  # <5% ROI
            recommendations.append({
                'type': 'warning',
                'category': 'Performance',
                'message': f'Low ROI of {roi:.1%}. Review strategy parameters.',
                'priority': 'high'
            })
            
        # Risk-based recommendations
        risk = self.consolidated_data['risk_assessment']
        max_drawdown = risk.get('max_drawdown', 0.0)
        
        if max_drawdown > 0.25:  # >25% drawdown
            recommendations.append({
                'type': 'warning',
                'category': 'Risk',
                'message': f'High drawdown of {max_drawdown:.1%}. Implement stricter risk controls.',
                'priority': 'high'
            })
        elif max_drawdown > 0.15:  # 15-25% drawdown
            recommendations.append({
                'type': 'neutral',
                'category': 'Risk',
                'message': f'Moderate drawdown of {max_drawdown:.1%}. Monitor risk exposure.',
                'priority': 'medium'
            })
            
        # Model validation recommendations
        validation = self.consolidated_data['model_validation']
        deployment_readiness = validation.get('deployment_readiness', 'Unknown')
        
        if deployment_readiness == 'Ready':
            recommendations.append({
                'type': 'positive',
                'category': 'Model',
                'message': 'Model validation passed. Ready for production deployment.',
                'priority': 'medium'
            })
        elif deployment_readiness == 'Caution':
            recommendations.append({
                'type': 'warning',
                'category': 'Model',
                'message': 'Model shows some concerns. Additional validation recommended.',
                'priority': 'high'
            })
            
        # Optimization recommendations
        optimization = self.consolidated_data['optimization']
        improvement_potential = optimization.get('improvement_potential', 0.0)
        
        if improvement_potential > 0.20:  # >20% improvement possible
            recommendations.append({
                'type': 'positive',
                'category': 'Optimization',
                'message': f'{improvement_potential:.1%} performance improvement possible through optimization.',
                'priority': 'high'
            })
            
        self.consolidated_data['recommendations'] = recommendations
        
    def collect_chart_links(self):
        """Collect references to all generated charts."""
        charts = []
        
        # Chart categories and their patterns
        chart_categories = {
            'Main Analysis': 'main_analysis_*.html',
            'Strategy Optimization': 'strategy_optimization_*.html', 
            'Risk Assessment': 'risk_assessment_*.html',
            'Model Validation': 'model_validation_*.html'
        }
        
        for category, pattern in chart_categories.items():
            chart_files = list(self.charts_dir.glob(pattern))
            
            for chart_file in chart_files:
                charts.append({
                    'category': category,
                    'filename': chart_file.name,
                    'title': self.generate_chart_title(chart_file.name),
                    'relative_path': f'all_charts/{chart_file.name}'
                })
                
        self.consolidated_data['charts'] = charts
        
    def generate_chart_title(self, filename: str) -> str:
        """Generate human-readable chart title from filename."""
        # Remove timestamp and extension
        name = filename.replace('.html', '')
        
        # Simple title generation
        title_map = {
            'main_analysis': 'Backtesting Overview',
            'strategy_optimization': 'Strategy Optimization',
            'risk_assessment': 'Risk Analysis', 
            'model_validation': 'Model Validation'
        }
        
        for key, title in title_map.items():
            if key in name:
                return title
                
        return name.replace('_', ' ').title()
        
    def generate_html_dashboard(self) -> str:
        """
        Generate consolidated HTML dashboard.
        
        Returns:
            str: Generated HTML content
        """
        # Get consolidated data
        overview = self.consolidated_data['overview']
        performance = self.consolidated_data['performance']
        optimization = self.consolidated_data['optimization']
        risk = self.consolidated_data['risk_assessment']
        validation = self.consolidated_data['model_validation']
        recommendations = self.consolidated_data['recommendations']
        charts = self.consolidated_data['charts']
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hockey Prediction System - Executive Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007acc;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .section h2 {{
            color: #007acc;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }}
        .recommendation {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .rec-positive {{ 
            background: #d4edda; 
            border-color: #28a745; 
        }}
        .rec-warning {{ 
            background: #fff3cd; 
            border-color: #ffc107; 
        }}
        .rec-neutral {{ 
            background: #e2e3e5; 
            border-color: #6c757d; 
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .chart-link {{
            padding: 15px;
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            text-decoration: none;
            color: #007acc;
            display: block;
            transition: background 0.3s;
        }}
        .chart-link:hover {{
            background: #e9ecef;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .summary-table th, .summary-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .summary-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏒 Hockey Prediction System</h1>
            <h2>Executive Analysis Dashboard</h2>
            <p>Automated Analysis Run - {overview.get('run_timestamp', 'N/A')}</p>
        </div>
        
        <!-- Key Metrics Overview -->
        <div class="section">
            <h2>📊 Key Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{performance.get('overall_roi', 0):.1%}</div>
                    <div class="metric-label">Overall ROI</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance.get('win_rate', 0):.1%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${performance.get('total_profit', 0):,.0f}</div>
                    <div class="metric-label">Total Profit</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance.get('model_accuracy', 0):.1%}</div>
                    <div class="metric-label">Model Accuracy</div>
                </div>
            </div>
        </div>
        
        <!-- Analysis Summary -->
        <div class="section">
            <h2>📋 Analysis Summary</h2>
            <table class="summary-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>Analysis Period</td>
                    <td>{overview.get('analysis_period', 'N/A')}</td>
                    <td>✅</td>
                </tr>
                <tr>
                    <td>Total Games Analyzed</td>
                    <td>{overview.get('total_games', 0):,}</td>
                    <td>✅</td>
                </tr>
                <tr>
                    <td>Number of Bets</td>
                    <td>{performance.get('number_of_bets', 0):,}</td>
                    <td>✅</td>
                </tr>
                <tr>
                    <td>Max Drawdown</td>
                    <td>{risk.get('max_drawdown', 0):.1%}</td>
                    <td>{'⚠️' if risk.get('max_drawdown', 0) > 0.20 else '✅'}</td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{risk.get('sharpe_ratio', 0):.2f}</td>
                    <td>{'✅' if risk.get('sharpe_ratio', 0) > 1.0 else '⚠️'}</td>
                </tr>
                <tr>
                    <td>Model Deployment</td>
                    <td>{validation.get('deployment_readiness', 'Unknown')}</td>
                    <td>{'✅' if validation.get('deployment_readiness') == 'Ready' else '⚠️'}</td>
                </tr>
            </table>
        </div>
        
        <!-- Recommendations -->
        <div class="section">
            <h2>💡 Business Recommendations</h2>
"""
        
        # Add recommendations
        if recommendations:
            for rec in recommendations:
                rec_class = f"rec-{rec['type']}"
                priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
                html_content += f"""
            <div class="recommendation {rec_class}">
                <strong>{priority_emoji} {rec['category']}</strong><br>
                {html.escape(rec['message'])}
            </div>
"""
        else:
            html_content += "<p>No specific recommendations generated.</p>"
            
        html_content += """
        </div>
        
        <!-- Strategy Optimization Results -->
        <div class="section">
            <h2>🎯 Strategy Optimization</h2>
"""
        
        if optimization.get('best_strategy'):
            html_content += f"""
            <p><strong>Best Strategy Identified:</strong> {optimization['best_strategy'].get('name', 'N/A')}</p>
            <p><strong>Improvement Potential:</strong> {optimization.get('improvement_potential', 0):.1%}</p>
"""
        
        html_content += """
        </div>
        
        <!-- Charts and Detailed Analysis -->
        <div class="section">
            <h2>📈 Detailed Charts & Analysis</h2>
            <div class="chart-grid">
"""
        
        # Add chart links
        for chart in charts:
            html_content += f"""
                <a href="{chart['relative_path']}" class="chart-link" target="_blank">
                    <strong>{chart['category']}</strong><br>
                    {chart['title']}
                </a>
"""
        
        html_content += f"""
            </div>
        </div>
        
        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Run ID: {overview.get('run_timestamp', 'N/A')} |
            Notebooks Executed: {overview.get('notebooks_executed', 0)}/4
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
        
    def generate_dashboard(self) -> str:
        """
        Main method to generate consolidated dashboard.
        
        Returns:
            str: Path to generated dashboard file
        """
        self.logger.info("🎨 Generating master dashboard...")
        
        # Step 1: Load individual summaries
        if not self.load_individual_summaries():
            self.logger.warning("⚠️ No summaries loaded - generating basic dashboard")
            
        # Step 2: Consolidate data from all sources
        self.consolidate_overview_data()
        self.consolidate_performance_data()
        self.consolidate_optimization_data()
        self.consolidate_risk_data()
        self.consolidate_validation_data()
        
        # Step 3: Generate recommendations
        self.generate_recommendations()
        
        # Step 4: Collect chart references
        self.collect_chart_links()
        
        # Step 5: Generate HTML dashboard
        html_content = self.generate_html_dashboard()
        
        # Step 6: Save dashboard file
        dashboard_file = self.run_dir / 'master_dashboard.html'
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        self.logger.info(f"✅ Master dashboard generated: {dashboard_file}")
        
        # Step 7: Save consolidated data as JSON
        json_file = self.run_dir / 'consolidated_summary.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.consolidated_data, f, indent=2, ensure_ascii=False, default=str)
            
        return str(dashboard_file)


def generate_master_dashboard(run_dir: str, run_timestamp: str) -> str:
    """
    Convenience function to generate master dashboard.
    
    Args:
        run_dir: Directory containing notebook outputs
        run_timestamp: Timestamp for this run
        
    Returns:
        str: Path to generated dashboard
    """
    generator = MasterDashboardGenerator(Path(run_dir), run_timestamp)
    return generator.generate_dashboard()


if __name__ == "__main__":
    # Example usage for testing
    import tempfile
    import json
    from pathlib import Path
    
    # Create temporary test structure
    test_dir = Path(tempfile.mkdtemp())
    summaries_dir = test_dir / 'individual_summaries'
    charts_dir = test_dir / 'all_charts'
    summaries_dir.mkdir(parents=True)
    charts_dir.mkdir(parents=True)
    
    # Create sample summary
    sample_summary = {
        'analysis_period': '2024/25 Season',
        'total_games': 1000,
        'overall_roi': 0.12,
        'win_rate': 0.58,
        'total_profit': 1200
    }
    
    with open(summaries_dir / 'main_analysis_summary_20250101_120000.json', 'w') as f:
        json.dump(sample_summary, f)
        
    # Generate dashboard
    generator = MasterDashboardGenerator(test_dir, '20250101_120000')
    dashboard_path = generator.generate_dashboard()
    
    print(f"🎨 Test dashboard generated: {dashboard_path}")
