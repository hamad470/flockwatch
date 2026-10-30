Attribute VB_Name = "WeeklyReportMacros"
' FlockWatch weekly report helpers.
' How to use: open a weekly_report_*.xlsx, press Alt+F11, File > Import File... > this .bas,
' then run FormatWeeklyReport (Alt+F8). Save as .xlsm if you want to keep the macros.
Option Explicit

Public Sub FormatWeeklyReport()
    Dim ws As Worksheet
    Application.ScreenUpdating = False
    For Each ws In ThisWorkbook.Worksheets
        ws.Cells.Font.Name = "Arial"
        ws.Cells.Font.Size = 10
        ws.UsedRange.Columns.AutoFit
        ColourAlertLevels ws
        SetPrintLayout ws
    Next ws
    AddAlertCountToSummary
    Application.ScreenUpdating = True
    MsgBox "Report formatted.", vbInformation, "FlockWatch"
End Sub

Private Sub ColourAlertLevels(ws As Worksheet)
    Dim headerRow As Range, cell As Range, col As Long, lastRow As Long, r As Long
    Set headerRow = ws.Rows(1)
    If ws.Name = "Summary" Then Set headerRow = ws.Rows(3)
    For Each cell In headerRow.Cells
        If cell.Value = "alert_level" Then col = cell.Column: Exit For
        If cell.Column > 30 Then Exit For
    Next cell
    If col = 0 Then Exit Sub
    lastRow = ws.Cells(ws.Rows.Count, col).End(xlUp).Row
    For r = headerRow.Row + 1 To lastRow
        Select Case ws.Cells(r, col).Value
            Case "Red": ws.Cells(r, col).Interior.Color = RGB(248, 215, 218)
            Case "Amber": ws.Cells(r, col).Interior.Color = RGB(255, 232, 204)
            Case "Green": ws.Cells(r, col).Interior.Color = RGB(212, 237, 218)
        End Select
    Next r
End Sub

Private Sub SetPrintLayout(ws As Worksheet)
    On Error Resume Next   ' printer settings can fail when no printer is installed
    With ws.PageSetup
        .Orientation = xlLandscape
        .Zoom = False
        .FitToPagesWide = 1
        .FitToPagesTall = False
        .CenterFooter = "FlockWatch - &A - page &P of &N"
    End With
    On Error GoTo 0
End Sub

Private Sub AddAlertCountToSummary()
    Dim ws As Worksheet, alerts As Worksheet, n As Long
    Set ws = ThisWorkbook.Worksheets("Summary")
    Set alerts = ThisWorkbook.Worksheets("Alerts")
    n = Application.WorksheetFunction.CountA(alerts.Range("A:A")) - 1
    ws.Range("I3").Value = "Alert days this week"
    ws.Range("I3").Font.Bold = True
    ws.Range("I4").Value = n
End Sub
