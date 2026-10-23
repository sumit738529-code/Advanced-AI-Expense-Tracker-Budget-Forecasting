"""PDF Statement Generator for SpendSense AI using ReportLab."""
import io
from datetime import datetime, date
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

def generate_pdf_statement(user, transactions, health, forecast, month_name=None):
    """Generates an executive monthly financial statement PDF in-memory."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#17223b'),
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#6d7890'),
        spaceAfter=14
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#17223b'),
        fontName='Helvetica-Bold',
        spaceBefore=10,
        spaceAfter=6
    )
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#19233b')
    )
    cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#19233b')
    )

    # Header
    today_str = datetime.now().strftime("%d %B %Y, %I:%M %p")
    period_str = month_name or date.today().strftime("%B %Y")
    
    story.append(Paragraph("✦ SpendSense AI — Monthly Financial Statement", title_style))
    story.append(Paragraph(f"Account Holder: <b>{user.name}</b> ({user.email}) &nbsp;|&nbsp; Statement Period: <b>{period_str}</b> &nbsp;|&nbsp; Generated: {today_str}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#4778ed'), spaceAfter=14))

    # KPI Summary Cards in Table format
    kpi_data = [
        [
            Paragraph("<b>Total Recorded Income</b>", cell_style),
            Paragraph("<b>Total Recorded Expenses</b>", cell_style),
            Paragraph("<b>Net Savings</b>", cell_style),
            Paragraph("<b>Financial Health</b>", cell_style)
        ],
        [
            Paragraph(f"<font size='12' color='#12a887'><b>₹{health['income']:,.2f}</b></font>", cell_style),
            Paragraph(f"<font size='12' color='#e16e79'><b>₹{health['expenses']:,.2f}</b></font>", cell_style),
            Paragraph(f"<font size='12' color='#4778ed'><b>₹{health['net_savings']:,.2f}</b></font>", cell_style),
            Paragraph(f"<font size='12' color='#17223b'><b>{health['score']}/100 ({health['grade']})</b></font>", cell_style)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[135, 135, 135, 135])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f6f8fc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#dfe4ef')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dfe4ef')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 14))

    # 50/30/20 Rule Analysis
    story.append(Paragraph("50/30/20 Financial Discipline Analysis", section_heading))
    rule_data = [
        [Paragraph("<b>Category</b>", cell_bold), Paragraph("<b>Target</b>", cell_bold), Paragraph("<b>Actual Spent</b>", cell_bold), Paragraph("<b>Share of Income</b>", cell_bold), Paragraph("<b>Status</b>", cell_bold)],
        [Paragraph("Needs (Housing, Utilities, Food)", cell_style), "50%", f"₹{health['needs_amount']:,.2f}", f"{health['needs_pct']:.1f}%", "Optimal" if health['needs_pct'] <= 55 else "Above Target"],
        [Paragraph("Wants (Shopping, Entertainment, Travel)", cell_style), "30%", f"₹{health['wants_amount']:,.2f}", f"{health['wants_pct']:.1f}%", "Optimal" if health['wants_pct'] <= 35 else "Above Target"],
        [Paragraph("Savings / Debt Paydown", cell_style), "20%", f"₹{health['net_savings']:,.2f}", f"{health['savings_pct']:.1f}%", "On Track" if health['savings_pct'] >= 15 else "Action Needed"]
    ]
    t_rule = Table(rule_data, colWidths=[180, 70, 100, 100, 90])
    t_rule.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eef3ff')),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e7ebf3')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_rule)
    story.append(Spacer(1, 14))

    # AI Forecast Section
    story.append(Paragraph(f"AI Trend & Next-Month Forecast ({forecast['next_month']})", section_heading))
    forecast_text = (
        f"Based on historical patterns, your projected spend for <b>{forecast['next_month']}</b> is "
        f"<b>₹{forecast['predicted_spend']:,.2f}</b> (90% Confidence Band: ₹{forecast['lower_bound']:,.2f} – ₹{forecast['upper_bound']:,.2f}). "
        f"Algorithm: {forecast['method']}. Historical Mean Absolute Error (MAE): ₹{forecast['mae']:,.2f}."
    )
    story.append(Paragraph(forecast_text, cell_style))
    story.append(Spacer(1, 14))

    # Transactions List Table
    story.append(Paragraph(f"Recorded Transactions Log ({len(transactions)} records)", section_heading))
    tx_headers = [
        Paragraph("<b>Date</b>", cell_bold),
        Paragraph("<b>Description</b>", cell_bold),
        Paragraph("<b>Category</b>", cell_bold),
        Paragraph("<b>AI Tag</b>", cell_bold),
        Paragraph("<b>Type</b>", cell_bold),
        Paragraph("<b>Amount</b>", cell_bold)
    ]
    tx_rows = [tx_headers]
    for tx in transactions[:35]: # Limit for single clean printable sheet
        kind_color = '#12a887' if tx.kind == 'income' else '#19233b'
        amt_str = f"+₹{tx.amount:,.2f}" if tx.kind == 'income' else f"−₹{tx.amount:,.2f}"
        tx_rows.append([
            Paragraph(tx.tx_date.strftime("%d-%m-%Y"), cell_style),
            Paragraph(tx.description[:32], cell_style),
            Paragraph(tx.category, cell_style),
            Paragraph(f"{tx.ai_category} ({tx.ai_confidence:.0%})", cell_style),
            Paragraph(tx.kind.title(), cell_style),
            Paragraph(f"<font color='{kind_color}'><b>{amt_str}</b></font>", cell_style)
        ])

    t_tx = Table(tx_rows, colWidths=[65, 175, 80, 95, 55, 70])
    t_tx.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eef3ff')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e7ebf3')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (5,0), (5,-1), 'RIGHT'),
    ]))
    story.append(t_tx)
    story.append(Spacer(1, 20))

    # Footer note
    footer_text = "This document is computer generated by SpendSense AI for academic research & personal financial tracking purposes."
    story.append(Paragraph(f"<font size='8' color='#8791a5'>{footer_text}</font>", subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
