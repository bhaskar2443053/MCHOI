# Measure every native equation in a deck with PowerPoint: width/height in inches per point of font size.
param([string]$Deck, [string]$Out)
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($Deck, $true, $false, $false)
$result = @{}
try {
  foreach ($s in $p.Slides) {
    foreach ($sh in $s.Shapes) {
      if ($sh.Name -like 'equation *' -and $sh.HasTextFrame) {
        $tr = $sh.TextFrame2.TextRange
        $size = $tr.Characters(1, 1).Font.Size
        $name = $sh.Name.Substring(9)
        $result[$name] = @{ w = [math]::Round($tr.BoundWidth / 72 / $size, 6); h = [math]::Round($tr.BoundHeight / 72 / $size, 6); pt = $size }
      }
    }
  }
} finally { $p.Close() }
[IO.File]::WriteAllText($Out, ($result | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))  # no BOM, for JSON.parse
"measured $($result.Count) equations"
