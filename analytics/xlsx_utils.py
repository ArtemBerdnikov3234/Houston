
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

FONT = "Arial"
HEADER_FILL = PatternFill("solid", start_color="D9E2F3")


def style_sheet(ws, widths: dict | None = None, default_width: int = 18, wrap: bool = True) -> None:
    
    widths = widths or {}
    for cell in ws[1]:
        cell.font = Font(name=FONT, bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(cell.column)].width = widths.get(cell.value, default_width)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name=FONT)
            cell.alignment = Alignment(wrap_text=wrap, vertical="top")
    ws.freeze_panes = "A2"
