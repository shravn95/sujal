from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image,
)


class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            leading=28,
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=6,
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#6b7280'),
            spaceAfter=24,
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1a1a2e'),
            spaceBefore=18,
            spaceAfter=10,
            borderWidth=0,
            borderPadding=0,
        ))
        self.styles.add(ParagraphStyle(
            name='SubSectionHeader',
            parent=self.styles['Heading3'],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#374151'),
            spaceBefore=12,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='BodyText2',
            parent=self.styles['Normal'],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor('#374151'),
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='SmallText',
            parent=self.styles['Normal'],
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#6b7280'),
            spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name='KPIValue',
            parent=self.styles['Normal'],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1a1a2e'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
        ))
        self.styles.add(ParagraphStyle(
            name='KPILabel',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_CENTER,
            fontName='Helvetica',
        ))

    def generate_report(
        self,
        dataset_summary: Dict[str, Any],
        cleaning_report: Dict[str, Any],
        charts: List[Dict[str, Any]],
        kpis: Dict[str, Any],
        forecast: Optional[Dict[str, Any]],
        query: Optional[str],
        report_text: Optional[str],
    ) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        story = []
        story.extend(self._build_cover_page(dataset_summary, query))
        story.extend(self._build_executive_summary(dataset_summary, cleaning_report, kpis, query, report_text))
        story.extend(self._build_kpi_section(kpis))
        story.extend(self._build_data_quality_section(dataset_summary, cleaning_report))
        
        # Ensure charts exist - generate fallback from dataset/KPI data if needed
        effective_charts = charts if charts and isinstance(charts, list) and len(charts) > 0 else self._generate_fallback_charts(dataset_summary, kpis)
        
        if effective_charts:
            story.append(PageBreak())
            story.append(Paragraph("Visualizations & Analytical Findings", self.styles['SectionHeader']))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#8b5cf6'), spaceAfter=16, spaceBefore=8))
            
            for i, chart in enumerate(effective_charts, 1):
                if not isinstance(chart, dict):
                    continue
                
                title = chart.get('title', f'Chart {i}')
                chart_type = chart.get('type', 'line').lower()
                insight = chart.get('insight_tooltip', '')
                
                # Chart header
                story.append(Paragraph(f"{i}. {title}", self.styles['SubSectionHeader']))
                story.append(Paragraph(f"<b>Chart Type:</b> {chart_type.capitalize()}", self.styles['SmallText']))
                
                # Structured insight sections
                if insight:
                    # Parse the deep insight into structured sections
                    insight_sections = self._parse_deep_insight(insight)
                    for section_title, section_content in insight_sections:
                        story.append(Paragraph(f"<b>{section_title}</b>", self.styles['SubSectionHeader']))
                        story.append(Paragraph(section_content, self.styles['BodyText2']))
                        story.append(Spacer(1, 4))
                
                # Chart image
                x_data, y_data = self._extract_chart_data(chart)
                if x_data and y_data:
                    chart_img = self.generate_chart_image(x_data, y_data, title, chart_type)
                    if chart_img:
                        story.append(chart_img)
                        story.append(Spacer(1, 10))
                
                # Data table (max 5 rows, rounded decimals)
                data = chart.get('data', [])
                if data:
                    display_data = data[:5]  # Max 5 rows
                    if display_data and isinstance(display_data[0], dict):
                        cols = list(display_data[0].keys()) if display_data else []
                        if cols:
                            # Clean column names for display
                            clean_cols = [c.replace('_', ' ').title() for c in cols]
                            table_data = [clean_cols]
                            for row in display_data:
                                clean_row = []
                                for c in cols:
                                    val = row.get(c, '')
                                    # Round decimal values
                                    if isinstance(val, float):
                                        clean_row.append(f"{val:.2f}")
                                    else:
                                        clean_row.append(str(val))
                                table_data.append(clean_row)
                            
                            max_cols = 5
                            if len(cols) > max_cols:
                                table_data = [row[:max_cols] for row in table_data]
                                table_data[0][-1] = '...'
                            
                            col_width = 7.0 * inch / len(table_data[0])
                            chart_table = Table(table_data, colWidths=[col_width] * len(table_data[0]))
                            chart_table.setStyle(TableStyle([
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                                ('FONTSIZE', (0, 0), (-1, -1), 8),
                                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
                                ('TOPPADDING', (0, 0), (-1, -1), 5),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                            ]))
                            story.append(chart_table)
                
                story.append(Spacer(1, 16))
                # Section divider
                if i < len(effective_charts):
                    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e5e7eb'), spaceAfter=12, spaceBefore=4))
        
        if forecast:
            story.append(PageBreak())
            story.extend(self._build_forecast_section(forecast))
        story.extend(self._build_footer())

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    def _parse_deep_insight(self, insight: str) -> List[tuple]:
        """Parse deep insight into structured sections for PDF."""
        sections = []
        if not insight:
            return sections
        
        # Split by common section markers
        parts = insight.split('. ')
        current_section = ""
        current_content = []
        
        section_markers = {
            "trend": "Key Visual Takeaway",
            "trend & pattern": "Key Visual Takeaway",
            "key visual takeaway": "Key Visual Takeaway",
            "extremes": "Extremes & Distribution",
            "extremes & distribution": "Extremes & Distribution",
            "extremes &": "Extremes & Distribution",
            "distribution": "Extremes & Distribution",
            "anomalies": "Anomalies & Outliers",
            "anomaly": "Anomalies & Outliers",
            "concentration": "Concentration Analysis",
            "distribution:": "Extremes & Distribution",
            "domain implication": "Business Impact",
            "business impact": "Business Impact",
            "domain implication:": "Business Impact",
            "business impact:": "Business Impact",
        }
        
        current_section_title = "Key Visual Takeaway"
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Check if this part starts a new section
            part_lower = part.lower()
            matched_section = None
            for marker, section in section_markers.items():
                if part_lower.startswith(marker) or marker in part_lower[:30]:
                    matched_section = section
                    break
            
            if matched_section and matched_section != current_section_title:
                if current_content:
                    sections.append((current_section_title, '. '.join(current_content) + '.'))
                current_section_title = matched_section
                current_content = [part]
            else:
                current_content.append(part)
        
        if current_content:
            sections.append((current_section_title, '. '.join(current_content) + '.'))
        
        # If no sections parsed, return whole insight as Key Visual Takeaway
        if not sections:
            return [("Key Visual Takeaway", insight)]
        
        return sections

    def _generate_fallback_charts(self, dataset_summary: Dict[str, Any], kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate default charts from dataset summary and KPI data when no charts provided."""
        fallback_charts = []
        
        # Chart 1: Data Quality Breakdown (bar chart)
        qb = dataset_summary.get('quality_score_breakdown', {})
        if qb:
            quality_data = [
                {'name': 'Completeness', 'value': qb.get('completeness', 0)},
                {'name': 'Consistency', 'value': qb.get('consistency', 0)},
                {'name': 'Type Confidence', 'value': qb.get('type_confidence', 0)},
            ]
            fallback_charts.append({
                'title': 'Data Quality Score Breakdown',
                'type': 'bar',
                'data': quality_data,
                'insight_tooltip': 'Quality dimensions scored across completeness, consistency, and type confidence.'
            })
        
        # Chart 2: KPI Comparison (bar chart)
        primary_kpi = kpis.get('primary_kpi', 'Primary KPI')
        secondary_kpi = kpis.get('secondary_kpi')
        if secondary_kpi:
            kpi_data = [
                {'name': primary_kpi, 'value': float(str(kpis.get('formatted_value', kpis.get('value', 0))).replace(',', '')) if str(kpis.get('formatted_value', kpis.get('value', 0))).replace(',', '').replace('.', '').isdigit() else 0},
                {'name': secondary_kpi, 'value': float(str(kpis.get('secondary_formatted_value', kpis.get('secondary_value', 0))).replace(',', '')) if str(kpis.get('secondary_formatted_value', kpis.get('secondary_value', 0))).replace(',', '').replace('.', '').isdigit() else 0},
            ]
            fallback_charts.append({
                'title': 'Key Performance Indicators Comparison',
                'type': 'bar',
                'data': kpi_data,
                'insight_tooltip': 'Primary vs Secondary KPI values for quick comparison.'
            })
        
        # Chart 3: Column Type Distribution (pie chart)
        col_types = dataset_summary.get('column_types', {})
        if col_types:
            type_data = [{'name': k, 'value': v} for k, v in col_types.items()]
            fallback_charts.append({
                'title': 'Dataset Column Type Distribution',
                'type': 'pie',
                'data': type_data,
                'insight_tooltip': 'Proportional breakdown of column data types in the cleaned dataset.'
            })
        
        # Chart 4: Numeric Column Stats (bar chart) - if numeric columns exist
        numeric_cols = dataset_summary.get('numeric_columns', [])
        if numeric_cols and len(numeric_cols) > 0:
            # Create sample distribution data
            sample_data = [{'name': col, 'value': 1} for col in numeric_cols[:8]]
            fallback_charts.append({
                'title': 'Numeric Dimensions Overview',
                'type': 'bar',
                'data': sample_data,
                'insight_tooltip': 'Available numeric dimensions in the dataset.'
            })
        
        return fallback_charts

    def _build_cover_page(self, summary: Dict[str, Any], query: Optional[str]) -> List:
        elements = []
        # Clean title banner at top with 20pt margin (no large spacer)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("AutoInsights", self.styles['CustomTitle']))
        elements.append(Paragraph("Executive Analysis Report", self.styles['CustomSubtitle']))
        elements.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#8b5cf6'), spaceAfter=20, spaceBefore=4))
        elements.append(Spacer(1, 8))

        filename = summary.get('filename', 'dataset')
        row_count = summary.get('row_count', 0)
        col_count = summary.get('column_count', 0)
        primary_kpi = summary.get('primary_kpi', 'N/A')
        quality_score = summary.get('quality_score', 100)

        cover_data = [
            ['Dataset', filename],
            ['Records Analyzed', f"{row_count:,}"],
            ['Dimensions & Measures', str(col_count)],
            ['Primary KPI', primary_kpi],
            ['Data Quality Score', f"{quality_score}%"],
            ['Report Generated', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ]
        if query:
            cover_data.append(['Analytical Query', query])

        cover_table = Table(cover_data, colWidths=[2.2 * inch, 4.3 * inch])
        cover_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#374151')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1a1a2e')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#e5e7eb')),
        ]))
        elements.append(cover_table)
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(
            "Generated by AutoInsights AI Analytics Platform<br/>Powered by NVIDIA Nemotron-3 Ultra 550B",
            ParagraphStyle('FooterNote', parent=self.styles['SmallText'], alignment=TA_CENTER, textColor=colors.HexColor('#9ca3af'))
        ))
        return elements

    def _build_executive_summary(
        self,
        summary: Dict[str, Any],
        cleaning: Dict[str, Any],
        kpis: Dict[str, Any],
        query: Optional[str],
        report_text: Optional[str],
    ) -> List:
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#8b5cf6'), spaceAfter=12))

        if report_text:
            for line in report_text.split('\n'):
                line = line.strip()
                if not line:
                    elements.append(Spacer(1, 6))
                    continue
                if line.startswith('### '):
                    elements.append(Paragraph(line[4:], self.styles['SubSectionHeader']))
                elif line.startswith('## '):
                    elements.append(Paragraph(line[3:], self.styles['SectionHeader']))
                elif line.startswith('# '):
                    elements.append(Paragraph(line[2:], self.styles['SectionHeader']))
                elif line.startswith('- **') and '**:' in line:
                    bold_end = line.find('**:')
                    bold_part = line[2:bold_end + 2]
                    rest = line[bold_end + 2:]
                    elements.append(Paragraph(f"<b>{bold_part}</b>{rest}", self.styles['BodyText2']))
                else:
                    elements.append(Paragraph(line, self.styles['BodyText2']))
        else:
            cleaned_rows = cleaning.get('cleaned_rows', summary.get('row_count', 0))
            primary_kpi = kpis.get('primary_kpi', summary.get('primary_kpi', 'primary metric'))
            formatted_value = kpis.get('formatted_value', 'N/A')
            filename = summary.get('filename', 'dataset')

            elements.append(Paragraph(
                f"This report presents an automated analysis of <b>{filename}</b> ({summary.get('filename', 'dataset')}), "
                f"containing <b>{cleaned_rows:,}</b> cleansed records across <b>{summary.get('column_count', 0)}</b> attributes. "
                f"The detected primary KPI is <b>{primary_kpi}</b> with a total aggregated volume of <b>{formatted_value}</b>.",
                self.styles['BodyText2']
            ))
            elements.append(Spacer(1, 10))
            if query:
                elements.append(Paragraph(f"<b>Analytical Question:</b> {query}", self.styles['BodyText2']))
                elements.append(Spacer(1, 10))
            elements.append(Paragraph(
                "All metrics, aggregations, and visual summaries are computed directly from the locally cleaned dataset "
                "using verified computational methods. AI-generated insights are grounded in actual data calculations.",
                self.styles['BodyText2']
            ))

        return elements

    def _build_kpi_section(self, kpis: Dict[str, Any]) -> List:
        elements = []
        elements.append(Paragraph("Key Performance Indicators", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#8b5cf6'), spaceAfter=16))

        primary_kpi = kpis.get('primary_kpi', 'Primary KPI')
        # Use formatted_value first, then value, then display_name from kpi_selection_details
        primary_value = kpis.get('formatted_value') or kpis.get('value', '—')
        data_quality = kpis.get('data_quality', 100)
        total_rows = kpis.get('total_rows', '—')

        kpi_data = [
            [
                Paragraph(str(primary_value), self.styles['KPIValue']),
                Paragraph(str(data_quality) + '%', self.styles['KPIValue']),
                Paragraph(str(total_rows), self.styles['KPIValue']),
            ],
            [
                Paragraph(f"PRIMARY KPI<br/>{primary_kpi}", self.styles['KPILabel']),
                Paragraph("DATA QUALITY", self.styles['KPILabel']),
                Paragraph("CLEANED RECORDS", self.styles['KPILabel']),
            ],
        ]
        if kpis.get('secondary_kpi'):
            sec_kpi = kpis['secondary_kpi']
            sec_value = kpis.get('secondary_formatted_value') or kpis.get('secondary_value', '—')
            kpi_data[0].insert(1, Paragraph(str(sec_value), self.styles['KPIValue']))
            kpi_data[1].insert(1, Paragraph(f"SECONDARY KPI<br/>{sec_kpi}", self.styles['KPILabel']))

        kpi_table = Table(kpi_data, colWidths=[1.8 * inch] * len(kpi_data[0]))
        kpi_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ffffff')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f9fafb')]),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 12))
        return elements

    def _build_data_quality_section(self, summary: Dict[str, Any], cleaning: Dict[str, Any]) -> List:
        elements = []
        elements.append(Paragraph("Data Quality & Cleaning Summary", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#8b5cf6'), spaceAfter=12))

        qb = summary.get('quality_score_breakdown', {})
        if qb:
            elements.append(Paragraph("Quality Score Breakdown", self.styles['SubSectionHeader']))
            qb_data = [
                ['Dimension', 'Score', 'Assessment'],
                ['Completeness', f"{qb.get('completeness', 0):.1f}%", self._quality_label(qb.get('completeness', 0))],
                ['Consistency', f"{qb.get('consistency', 0):.1f}%", self._quality_label(qb.get('consistency', 0))],
                ['Type Confidence', f"{qb.get('type_confidence', 0):.1f}%", self._quality_label(qb.get('type_confidence', 0))],
                ['Overall', f"{summary.get('quality_score', 100):.1f}%", self._quality_label(summary.get('quality_score', 100))],
            ]
            qb_table = Table(qb_data, colWidths=[2.0 * inch, 1.2 * inch, 3.3 * inch])
            qb_table.setStyle(self._default_table_style())
            elements.append(qb_table)
            elements.append(Spacer(1, 12))

        cs = cleaning.get('cleaning_summary') or summary.get('cleaning_summary')
        if cs:
            elements.append(Paragraph("Cleaning Operations", self.styles['SubSectionHeader']))
            clean_rows = []
            for detail in cs.get('imputation_details', []):
                clean_rows.append(['Missing Value Imputation', detail])
            for detail in cs.get('high_missing_flagged', []):
                clean_rows.append(['High Missing Flagged', detail])
            for detail in cs.get('outlier_treatment_details', []):
                clean_rows.append(['Outlier Treatment', detail])
            if clean_rows:
                clean_data = [['Operation', 'Details']] + clean_rows
                clean_table = Table(clean_data, colWidths=[1.8 * inch, 4.7 * inch])
                clean_table.setStyle(self._default_table_style())
                elements.append(clean_table)
                elements.append(Spacer(1, 12))

            dup = cs.get('duplicates_removed', 0)
            if dup:
                elements.append(Paragraph(f"<b>Duplicate Rows Removed:</b> {dup:,}", self.styles['BodyText2']))
                elements.append(Spacer(1, 6))

        return elements

    def _quality_label(self, score: float) -> str:
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        return "Needs Review"

    def _build_forecast_section(self, forecast: Dict[str, Any]) -> List:
        elements = []
        elements.append(Paragraph("Forecasting & Projections", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#8b5cf6'), spaceAfter=12))

        metric = forecast.get('metric_name', 'Metric')
        periods = forecast.get('periods', 4)
        growth = forecast.get('projected_growth_rate', 'N/A')

        elements.append(Paragraph(f"<b>Target Metric:</b> {metric}", self.styles['BodyText2']))
        elements.append(Paragraph(f"<b>Forecast Horizon:</b> {periods} periods", self.styles['BodyText2']))
        elements.append(Paragraph(f"<b>Projected Growth Rate:</b> {growth}", self.styles['BodyText2']))
        elements.append(Spacer(1, 10))

        # Add forecast visualization
        forecast_img = self._create_forecast_chart_image(forecast)
        if forecast_img:
            elements.append(forecast_img)
            elements.append(Spacer(1, 10))

        best = forecast.get('best_case', [])
        expected = forecast.get('expected_case', [])
        worst = forecast.get('worst_case', [])

        if best or expected or worst:
            elements.append(Paragraph("Scenario Projections", self.styles['SubSectionHeader']))
            max_len = max(len(best), len(expected), len(worst))
            fc_data = [['Period', 'Best Case', 'Expected', 'Worst Case']]
            for i in range(max_len):
                period_name = ''
                if best and i < len(best) and isinstance(best[i], dict):
                    period_name = best[i].get('period', f'Period {i+1}')
                elif expected and i < len(expected) and isinstance(expected[i], dict):
                    period_name = expected[i].get('period', f'Period {i+1}')
                elif worst and i < len(worst) and isinstance(worst[i], dict):
                    period_name = worst[i].get('period', f'Period {i+1}')
                else:
                    period_name = f'Period {i+1}'

                b_val = best[i].get('value', '—') if best and i < len(best) and isinstance(best[i], dict) else '—'
                e_val = expected[i].get('value', '—') if expected and i < len(expected) and isinstance(expected[i], dict) else '—'
                w_val = worst[i].get('value', '—') if worst and i < len(worst) and isinstance(worst[i], dict) else '—'

                fc_data.append([period_name, str(b_val), str(e_val), str(w_val)])

            fc_table = Table(fc_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
            fc_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            elements.append(fc_table)

        model_info = forecast.get('model_info', {})
        if model_info:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("<b>Model Details:</b>", self.styles['SmallText']))
            for k, v in model_info.items():
                elements.append(Paragraph(f"  • {k}: {v}", self.styles['SmallText']))

        return elements

    def _build_footer(self) -> List:
        elements = []
        elements.append(Spacer(1, 12))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e5e7eb'), spaceAfter=8))
        elements.append(Paragraph(
            "This report was generated by the AutoInsights AI Analytics Platform. "
            "All computations are performed locally on the cleaned dataset. "
            "AI-generated narratives are grounded in verified data calculations.",
            ParagraphStyle('Disclaimer', parent=self.styles['SmallText'], alignment=TA_CENTER, textColor=colors.HexColor('#9ca3af'))
        ))
        elements.append(Paragraph(
            f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            ParagraphStyle('Timestamp', parent=self.styles['SmallText'], alignment=TA_CENTER, textColor=colors.HexColor('#9ca3af'))
        ))
        return elements

    def _default_table_style(self) -> TableStyle:
        return TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ])

    def _extract_chart_data(self, chart: Dict[str, Any]) -> tuple:
        """Extract x and y data from various chart data formats."""
        data = chart.get('data', [])
        if not data:
            return [], []
        
        x_values = []
        y_values = []
        
        first_item = data[0] if data else {}
        
        if isinstance(first_item, dict):
            keys = set(first_item.keys())
            
            if 'x' in keys and 'y' in keys:
                x_values = [row.get('x', '') for row in data]
                y_values = [row.get('y', 0) for row in data]
            elif 'name' in keys and 'value' in keys:
                x_values = [row.get('name', '') for row in data]
                y_values = [row.get('value', 0) for row in data]
            elif 'label' in keys and 'value' in keys:
                x_values = [row.get('label', '') for row in data]
                y_values = [row.get('value', 0) for row in data]
            elif 'category' in keys and 'value' in keys:
                x_values = [row.get('category', '') for row in data]
                y_values = [row.get('value', 0) for row in data]
            else:
                all_keys = list(keys)
                if len(all_keys) >= 2:
                    x_values = [row.get(all_keys[0], '') for row in data]
                    y_values = [row.get(all_keys[1], 0) for row in data]
        
        elif isinstance(first_item, list) and len(first_item) >= 2:
            x_values = [row[0] for row in data]
            y_values = [row[1] for row in data]
        
        numeric_y = []
        for v in y_values:
            try:
                numeric_y.append(float(v))
            except (ValueError, TypeError):
                numeric_y.append(0)
        y_values = numeric_y
        
        return x_values, y_values

    def generate_chart_image(self, x_data: List, y_data: List, title: str = '', 
                            chart_type: str = 'line', 
                            width: float = 6.5 * inch, height: float = 3.5 * inch) -> Optional[Image]:
        """Create a matplotlib chart from x/y data arrays and return as reportlab Image."""
        if not x_data or not y_data or len(x_data) != len(y_data):
            return None
        
        # Normalize chart type
        chart_type = str(chart_type).lower().strip()
        
        # Vibrant color palette for bar/histogram charts
        bar_palette = ['#3B82F6', '#8B5CF6', '#10B981', '#EC4899', '#F59E0B', '#06B6D4', '#84CC16', '#F97316']
        
        try:
            fig, ax = plt.subplots(figsize=(width / 72, height / 72), dpi=100)
            fig.patch.set_facecolor('white')
            
            x_indices = range(len(x_data))
            x_labels = [str(x)[:15] for x in x_data]
            
            # Bar / Histogram
            if chart_type in ('bar', 'bar chart', 'histogram'):
                colors_list = [bar_palette[i % len(bar_palette)] for i in range(len(x_data))]
                ax.bar(x_indices, y_data, color=colors_list, alpha=0.85, edgecolor='white', linewidth=0.5)
            
            # Scatterplot
            elif chart_type in ('scatter', 'scatterplot'):
                ax.scatter(x_indices, y_data, color='#8B5CF6', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
            
            # Donut / Pie
            elif chart_type in ('pie', 'donut', 'donut chart'):
                colors_list = ['#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899', '#6366f1', '#14b8a6']
                is_donut = chart_type in ('donut', 'donut chart')
                
                # Calculate percentages to determine which slices get labels
                total = sum(y_data)
                percentages = [v / total * 100 for v in y_data]
                
                # Only show percentage labels for slices >= 10%
                autopct_labels = []
                for pct in percentages:
                    if pct >= 10:
                        autopct_labels.append(f'{pct:.1f}%')
                    else:
                        autopct_labels.append('')
                
                wedges, texts, autotexts = ax.pie(
                    y_data, 
                    labels=None,  # No labels on slices - use legend instead
                    autopct=lambda pct: f'{pct:.1f}%' if pct >= 10 else '',
                    colors=colors_list[:len(y_data)], 
                    startangle=90,
                    textprops={'fontsize': 8, 'fontweight': 'bold', 'color': 'white'},
                    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=1.5) if is_donut else dict(width=1.0, edgecolor='white', linewidth=1.5),
                    pctdistance=0.75 if is_donut else 0.6,
                )
                
                # Clean up autotexts - only show for >= 10%
                for i, autotext in enumerate(autotexts):
                    if percentages[i] >= 10:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(8)
                    else:
                        autotext.set_text('')  # Hide small slice labels
                
                # Add clean legend at bottom
                ax.legend(wedges, x_labels, 
                         title="Categories",
                         loc="lower center",
                         bbox_to_anchor=(0.5, -0.15),
                         ncol=3,
                         fontsize=8,
                         frameon=False,
                         title_fontsize=9)
            
            # Line / Area / Trend
            else:
                ax.plot(x_indices, y_data, color='#8B5CF6', linewidth=2.5, marker='o', markersize=5, 
                       markerfacecolor='white', markeredgewidth=2, markeredgecolor='#8B5CF6')
                if chart_type in ('area', 'trend'):
                    ax.fill_between(x_indices, y_data, alpha=0.12, color='#8B5CF6')
            
            # X-axis configuration for non-pie charts
            if chart_type not in ('pie', 'donut', 'donut chart'):
                # Limit tick frequency if too many items
                if len(x_data) > 10:
                    step = max(1, len(x_data) // 10)
                    tick_positions = list(x_indices)[::step]
                    tick_labels = x_labels[::step]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
                else:
                    ax.set_xticks(x_indices)
                    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
            
            if title:
                ax.set_title(title, fontsize=11, fontweight='bold', color='#1a1a2e', pad=12)
            
            if chart_type not in ('pie', 'donut', 'donut chart'):
                ax.set_facecolor('#fafafa')
                ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#d1d5db')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#e5e7eb')
                ax.spines['bottom'].set_color('#e5e7eb')
                ax.tick_params(axis='y', labelsize=8, colors='#6b7280')
                ax.tick_params(axis='x', labelsize=8, colors='#6b7280')
            else:
                # For pie/donut, ensure clean look
                ax.set_facecolor('#fafafa')
            
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='PNG', dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            img_buffer.seek(0)
            
            return Image(img_buffer, width=width, height=height)
        except Exception:
            return None

    def _create_chart_image(self, chart_data: List[Dict], x_key: str, y_key: str, 
                           chart_type: str = 'line', title: str = '', 
                           width: float = 6.5 * inch, height: float = 3.5 * inch) -> Optional[Image]:
        """Create a matplotlib chart and return as reportlab Image (legacy method)."""
        if not chart_data:
            return None
        
        try:
            x_values = [row.get(x_key, '') for row in chart_data]
            y_values = [row.get(y_key, 0) for row in chart_data]
            
            numeric_y = []
            for v in y_values:
                try:
                    numeric_y.append(float(v))
                except (ValueError, TypeError):
                    numeric_y.append(0)
            y_values = numeric_y
            
            return self.generate_chart_image(x_values, y_values, title, chart_type, width, height)
        except Exception:
            return None

    def _create_forecast_chart_image(self, forecast: Dict[str, Any], 
                                     width: float = 6.5 * inch, height: float = 3.5 * inch) -> Optional[Image]:
        """Create a forecast chart with best/expected/worst case scenarios."""
        best = forecast.get('best_case', [])
        expected = forecast.get('expected_case', [])
        worst = forecast.get('worst_case', [])
        
        if not (best or expected or worst):
            return None
        
        try:
            max_len = max(len(best), len(expected), len(worst))
            periods = []
            best_vals = []
            expected_vals = []
            worst_vals = []
            
            for i in range(max_len):
                if best and i < len(best) and isinstance(best[i], dict):
                    periods.append(str(best[i].get('period', f'P{i+1}'))[:10])
                    best_vals.append(float(best[i].get('value', 0)))
                else:
                    periods.append(f'P{i+1}')
                    best_vals.append(None)
                
                if expected and i < len(expected) and isinstance(expected[i], dict):
                    expected_vals.append(float(expected[i].get('value', 0)))
                else:
                    expected_vals.append(None)
                
                if worst and i < len(worst) and isinstance(worst[i], dict):
                    worst_vals.append(float(worst[i].get('value', 0)))
                else:
                    worst_vals.append(None)
            
            fig, ax = plt.subplots(figsize=(width / 72, height / 72), dpi=100)
            fig.patch.set_facecolor('white')
            
            x = range(len(periods))
            
            # Plot expected case
            valid_expected = [(i, v) for i, v in enumerate(expected_vals) if v is not None]
            if valid_expected:
                xe, ye = zip(*valid_expected)
                ax.plot(xe, ye, color='#8b5cf6', linewidth=2.5, marker='o', markersize=5, 
                       label='Expected', markerfacecolor='white', markeredgewidth=2)
            
            # Plot best case
            valid_best = [(i, v) for i, v in enumerate(best_vals) if v is not None]
            if valid_best:
                xb, yb = zip(*valid_best)
                ax.plot(xb, yb, color='#10b981', linewidth=2, linestyle='--', marker='^', markersize=4,
                       label='Best Case', markerfacecolor='white', markeredgewidth=1.5)
            
            # Plot worst case
            valid_worst = [(i, v) for i, v in enumerate(worst_vals) if v is not None]
            if valid_worst:
                xw, yw = zip(*valid_worst)
                ax.plot(xw, yw, color='#ef4444', linewidth=2, linestyle='--', marker='v', markersize=4,
                       label='Worst Case', markerfacecolor='white', markeredgewidth=1.5)
            
            # Fill between best and worst
            if valid_best and valid_worst:
                min_len = min(len(valid_best), len(valid_worst))
                ax.fill_between([vb[0] for vb in valid_best[:min_len]], 
                               [vb[1] for vb in valid_best[:min_len]],
                               [vw[1] for vw in valid_worst[:min_len]],
                               alpha=0.1, color='#8b5cf6')
            
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels(periods, rotation=45, ha='right', fontsize=8)
            ax.set_title('Forecast Scenarios', fontsize=11, fontweight='bold', color='#1a1a2e', pad=12)
            
            ax.set_facecolor('#fafafa')
            ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='#d1d5db')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#e5e7eb')
            ax.spines['bottom'].set_color('#e5e7eb')
            ax.tick_params(axis='y', labelsize=8, colors='#6b7280')
            ax.tick_params(axis='x', labelsize=8, colors='#6b7280')
            ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
            
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='PNG', dpi=100, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            img_buffer.seek(0)
            
            return Image(img_buffer, width=width, height=height)
        except Exception:
            return None


pdf_generator = PDFReportGenerator()