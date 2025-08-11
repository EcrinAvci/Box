public class Position
{
    public int X, Y, Z;
    public Position(int x, int y, int z) { X = x; Y = y; Z = z; }
}

public class Box
{
    public int Width, Height, Depth;
    public float Weight;
    public Box(int w, int h, int d, float weight) { Width = w; Height = h; Depth = d; Weight = weight; }
}

public class RotatedBox
{
    public Box Box { get; set; }
    public string Rotation { get; set; }
    
    public RotatedBox(Box box, string rotation)
    {
        Box = box;
        Rotation = rotation;
    }
}

public class PlacementResult
{
    public Position Position { get; set; }
    public Box Box { get; set; }
    public string Rotation { get; set; }
}

public class Container
{
    public int Width, Height, Depth;
    private bool[,,] occupied;
    private List<PlacedBoxInfo> placedBoxes; // Yerleştirilen kutuların bilgilerini tut
    
    public Container(int w, int h, int d)
    {
        Width = w; Height = h; Depth = d;
        occupied = new bool[w, h, d];
        placedBoxes = new List<PlacedBoxInfo>();
    }
    
    public bool CanPlaceBox(int x, int y, int z, int w, int h, int d, int margin = 0)
    {
        // 1. Sınır kontrolü
        if (x < 0 || y < 0 || z < 0 || x + w > Width || y + h > Height || z + d > Depth)
            return false;
        
        // 2. Koordinat bazında çakışma kontrolü
        for (int i = x; i < x + w; i++)
            for (int j = y; j < y + h; j++)
                for (int k = z; k < z + d; k++)
                    if (occupied[i, j, k]) 
                        return false;
        // 3. Kutu sınırları kontrolü (AABB çakışma)
        foreach (var existingBox in placedBoxes)
        {
            if (AabbOverlap(x, y, z, w, h, d, existingBox))
                return false;
        }
        return true;
    }

    public void PlaceBox(int x, int y, int z, int w, int h, int d, int margin = 0)
    {
        // Ekstra güvenlik: Çakışma kontrolü
        foreach (var existingBox in placedBoxes)
        {
            if (AabbOverlap(x, y, z, w, h, d, existingBox))
            {
                Console.WriteLine($"❌ [Yerleştirme] Kutu çakışması! YERLEŞTİRİLMEDİ!");
                return;
            }
        }
        // Koordinatları işaretle
        for (int i = x; i < x + w; i++)
            for (int j = y; j < y + h; j++)
                for (int k = z; k < z + d; k++)
                    occupied[i, j, k] = true;
        // Kutu bilgisini kaydet
        placedBoxes.Add(new PlacedBoxInfo(x, y, z, w, h, d));
        Console.WriteLine($"✅ Kutu yerleştirildi: ({x},{y},{z},{w},{h},{d})");
    }

    // AABB çakışma kontrolü
    private bool AabbOverlap(int x1, int y1, int z1, int w1, int h1, int d1, PlacedBoxInfo b2)
    {
        bool overlapX = x1 < b2.X + b2.Width && x1 + w1 > b2.X;
        bool overlapY = y1 < b2.Y + b2.Height && y1 + h1 > b2.Y;
        bool overlapZ = z1 < b2.Z + b2.Depth && z1 + d1 > b2.Z;
        return overlapX && overlapY && overlapZ;
    }
    
    // Marjinli yerleştirme fonksiyonu (daha esnek)
    public bool CanPlaceBoxWithMargin(int x, int y, int z, int w, int h, int d, int margin = 1)
    {
        return CanPlaceBox(x - margin, y - margin, z - margin, w + 2 * margin, h + 2 * margin, d + 2 * margin);
    }
    
    public void PlaceBoxWithMargin(int x, int y, int z, int w, int h, int d, int margin = 1)
    {
        PlaceBox(x - margin, y - margin, z - margin, w + 2 * margin, h + 2 * margin, d + 2 * margin);
    }
    
    // Yerleştirilen kutuların sayısını döndür
    public int GetPlacedBoxCount()
    {
        return placedBoxes.Count;
    }
    
    // Yerleştirilen kutuların bilgilerini döndür
    public List<PlacedBoxInfo> GetPlacedBoxes()
    {
        return placedBoxes.ToList();
    }
}

// Yerleştirilen kutu bilgilerini tutmak için yeni sınıf
public class PlacedBoxInfo
{
    public int X, Y, Z, Width, Height, Depth;
    
    public PlacedBoxInfo(int x, int y, int z, int w, int h, int d)
    {
        X = x; Y = y; Z = z; Width = w; Height = h; Depth = d;
    }
}

public class PlacedBox
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Z { get; set; }
    public int posX { get; set; }
    public int posY { get; set; }
    public int posZ { get; set; }
    public float Weight { get; set; }
    public float Volume { get; set; }
    public string Rotation { get; set; } = "XYZ";
} 