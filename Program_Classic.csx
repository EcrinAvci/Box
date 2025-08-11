using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text.Json;

class Program
{
    static void Main()
    {
        var mlContext = new MLContext();

        // JSON oku
        var json = File.ReadAllText("data.json");
        var dataList = JsonSerializer.Deserialize<List<BoxData>>(json)!;
        var data = mlContext.Data.LoadFromEnumerable(dataList);

        // Ortak özellikler
        var featureColumns = new[] { "X", "Y", "Z", "Weight" };

        // --- posX modeli ---
        var pipelineX = mlContext.Transforms.Concatenate("Features", featureColumns)
            .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: nameof(BoxData.posX)));

        var modelX = pipelineX.Fit(data);
        var predEngineX = mlContext.Model.CreatePredictionEngine<BoxData, PosPrediction>(modelX);

        // --- posY modeli ---
        var pipelineY = mlContext.Transforms.Concatenate("Features", featureColumns)
            .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: nameof(BoxData.posY)));

        var modelY = pipelineY.Fit(data);
        var predEngineY = mlContext.Model.CreatePredictionEngine<BoxData, PosPrediction>(modelY);

        // --- posZ modeli ---
        var pipelineZ = mlContext.Transforms.Concatenate("Features", featureColumns)
            .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: nameof(BoxData.posZ)));

        var modelZ = pipelineZ.Fit(data);
        var predEngineZ = mlContext.Model.CreatePredictionEngine<BoxData, PosPrediction>(modelZ);

        // Kutu yerleştirme algoritması (rotasyon ile)
        var container = new Container(100, 100, 100);
        var placedBoxes = new List<PlacedBox>();
        var totalVolume = 0.0;
        var placedCount = 0;

        // Kutuları daha akıllı sırala
        var sortedBoxes = dataList
            .OrderByDescending(b => b.Volume) // Önce büyük kutular
            .ThenByDescending(b => Math.Max(b.X, Math.Max(b.Y, b.Z))) // En uzun kenar
            .ThenByDescending(b => b.Weight) // Ağır kutular önce
            .ToList();

        Console.WriteLine("Kutu yerleştirme başlıyor (gelişmiş algoritma ile)...");
        Console.WriteLine($"Toplam {sortedBoxes.Count} kutu işlenecek\n");

        foreach (var boxData in sortedBoxes)
        {
            var box = new Box((int)boxData.X, (int)boxData.Y, (int)boxData.Z, boxData.Weight);
            
            // Gelişmiş rotasyon ile optimal yerleştirme
            var placement = FindOptimalPositionWithRotation(container, box);
            
            if (placement != null)
            {
                var placedBox = new PlacedBox
                {
                    X = placement.Box.Width,
                    Y = placement.Box.Height,
                    Z = placement.Box.Depth,
                    posX = placement.Position.X,
                    posY = placement.Position.Y,
                    posZ = placement.Position.Z,
                    Weight = box.Weight,
                    Volume = placement.Box.Width * placement.Box.Height * placement.Box.Depth,
                    Rotation = placement.Rotation
                };
                
                placedBoxes.Add(placedBox);
                totalVolume += placedBox.Volume;
                placedCount++;
                
                // Konteyneri güncelle
                container.PlaceBox(placement.Position.X, placement.Position.Y, placement.Position.Z, 
                                 placement.Box.Width, placement.Box.Height, placement.Box.Depth);
                
                // Detaylı bilgi yazdır
                var originalVolume = boxData.X * boxData.Y * boxData.Z;
                var efficiency = (placedBox.Volume / originalVolume) * 100;
                Console.WriteLine($"Kutu {placedCount}: {boxData.X}x{boxData.Y}x{boxData.Z} -> {placement.Box.Width}x{placement.Box.Height}x{placement.Box.Depth} (Rotasyon: {placement.Rotation}) | Verimlilik: {efficiency:F1}% | Pozisyon: ({placement.Position.X},{placement.Position.Y},{placement.Position.Z})");
            }
            else
            {
                Console.WriteLine($"❌ Kutu yerleştirilemedi: {boxData.X}x{boxData.Y}x{boxData.Z} (Hacim: {boxData.Volume})");
            }
        }

        // Sonuçları yazdır
        Console.WriteLine($"\nToplam {placedCount} kutu yerleştirildi");
        Console.WriteLine($"Toplam hacim: {totalVolume:F2}");
        Console.WriteLine($"Doluluk oranı: {(totalVolume / (100 * 100 * 100) * 100):F2}%");

        // Yeni kutu ekleme örneği (rotasyon ile)
        Console.WriteLine("\n--- Yeni Kutu Ekleme Testi (Rotasyon ile) ---");
        var newBox = new BoxData { X = 8, Y = 6, Z = 4, Weight = 2.5f, Volume = 192 };
        
        // ML modeli ile yeni kutunun pozisyonunu tahmin et
        var predX = predEngineX.Predict(newBox).Value;
        var predY = predEngineY.Predict(newBox).Value;
        var predZ = predEngineZ.Predict(newBox).Value;
        
        Console.WriteLine($"Yeni kutu boyutları: {newBox.X} x {newBox.Y} x {newBox.Z}");
        Console.WriteLine($"ML tahmini pozisyon: X:{predX:F2}, Y:{predY:F2}, Z:{predZ:F2}");
        
        // Rotasyon ile en iyi pozisyonu bul
        var newPlacement = FindBestPositionWithRotation(container, new Box((int)newBox.X, (int)newBox.Y, (int)newBox.Z, newBox.Weight), (int)predX, (int)predY, (int)predZ);
        
        if (newPlacement != null)
        {
            var newPlacedBox = new PlacedBox
            {
                X = newPlacement.Box.Width,
                Y = newPlacement.Box.Height,
                Z = newPlacement.Box.Depth,
                posX = newPlacement.Position.X,
                posY = newPlacement.Position.Y,
                posZ = newPlacement.Position.Z,
                Weight = newBox.Weight,
                Volume = newBox.Volume,
                Rotation = newPlacement.Rotation
            };
            
            placedBoxes.Add(newPlacedBox);
            totalVolume += newPlacedBox.Volume;
            placedCount++;
            
            Console.WriteLine($"Yeni kutu yerleştirildi! Boyutlar: {newPlacement.Box.Width}x{newPlacement.Box.Height}x{newPlacement.Box.Depth} (Rotasyon: {newPlacement.Rotation})");
            Console.WriteLine($"Pozisyon: X:{newPlacement.Position.X}, Y:{newPlacement.Position.Y}, Z:{newPlacement.Position.Z}");
            Console.WriteLine($"Güncel doluluk oranı: {(totalVolume / (100 * 100 * 100) * 100):F2}%");
        }
        else
        {
            Console.WriteLine("Yeni kutu için yer bulunamadı!");
        }

        // JSON dosyasına kaydet
        var result = new
        {
            container = new { width = 100, height = 100, depth = 100 },
            boxes = placedBoxes
        };

        var jsonResult = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText("project/placement_result.json", jsonResult);
        Console.WriteLine("Sonuçlar project/placement_result.json dosyasına kaydedildi");
    }



    static PlacementResult? FindOptimalPositionWithRotation(Container container, Box originalBox)
    {
        var bestPlacement = (PlacementResult?)null;
        var bestScore = double.MinValue;

        // Tüm olası rotasyonları dene
        var rotations = GetBoxRotations(originalBox);
        
        foreach (var rotatedBox in rotations)
        {
            var position = FindOptimalPosition(container, rotatedBox.Box);
            if (position != null)
            {
                // Skor hesapla (daha düşük Z koordinatı ve daha iyi sığma)
                var score = CalculatePlacementScore(position, rotatedBox.Box, container);
                
                if (score > bestScore)
                {
                    bestScore = score;
                    bestPlacement = new PlacementResult
                    {
                        Position = position,
                        Box = rotatedBox.Box,
                        Rotation = rotatedBox.Rotation
                    };
                }
            }
        }
        
        return bestPlacement;
    }

    static PlacementResult? FindBestPositionWithRotation(Container container, Box originalBox, int predX, int predY, int predZ)
    {
        var bestPlacement = (PlacementResult?)null;
        var bestScore = double.MinValue;

        // Tüm olası rotasyonları dene
        var rotations = GetBoxRotations(originalBox);
        
        foreach (var rotatedBox in rotations)
        {
            var position = FindBestPosition(container, rotatedBox.Box, predX, predY, predZ);
            if (position != null)
            {
                // Skor hesapla (tahmin edilen pozisyona yakınlık + sığma kalitesi)
                var score = CalculatePlacementScore(position, rotatedBox.Box, container, predX, predY, predZ);
                
                if (score > bestScore)
                {
                    bestScore = score;
                    bestPlacement = new PlacementResult
                    {
                        Position = position,
                        Box = rotatedBox.Box,
                        Rotation = rotatedBox.Rotation
                    };
                }
            }
        }
        
        return bestPlacement;
    }

    static List<RotatedBox> GetBoxRotations(Box originalBox)
    {
        var rotations = new List<RotatedBox>();
        
        // Orijinal boyutlar
        rotations.Add(new RotatedBox(originalBox, "XYZ"));
        
        // X ekseni etrafında 90° döndür
        rotations.Add(new RotatedBox(new Box(originalBox.Width, originalBox.Depth, originalBox.Height, originalBox.Weight), "XZY"));
        
        // Y ekseni etrafında 90° döndür
        rotations.Add(new RotatedBox(new Box(originalBox.Depth, originalBox.Height, originalBox.Width, originalBox.Weight), "ZYX"));
        
        // Z ekseni etrafında 90° döndür
        rotations.Add(new RotatedBox(new Box(originalBox.Height, originalBox.Width, originalBox.Depth, originalBox.Weight), "YXZ"));
        
        // X ve Y ekseni etrafında 90° döndür
        rotations.Add(new RotatedBox(new Box(originalBox.Depth, originalBox.Width, originalBox.Height, originalBox.Weight), "ZXY"));
        
        // Y ve Z ekseni etrafında 90° döndür
        rotations.Add(new RotatedBox(new Box(originalBox.Height, originalBox.Depth, originalBox.Width, originalBox.Weight), "YZX"));
        
        return rotations;
    }

    static double CalculatePlacementScore(Position position, Box box, Container container, int? predX = null, int? predY = null, int? predZ = null)
    {
        var score = 0.0;
        
        // 1. Yerçekimi etkisi - daha düşük Z koordinatı tercih edilir
        score -= position.Z * 15;
        
        // 2. Alt kısım tercihi - daha düşük Y koordinatı
        score -= position.Y * 8;
        
        // 3. Sol taraf tercihi - daha düşük X koordinatı
        score -= position.X * 3;
        
        // 4. Boşluk analizi - etrafındaki boşlukları kontrol et
        var spaceEfficiency = CalculateSpaceEfficiency(position, box, container);
        score += spaceEfficiency * 50;
        
        // 5. Eğer tahmin edilen pozisyon varsa, yakınlık bonusu
        if (predX.HasValue && predY.HasValue && predZ.HasValue)
        {
            var distance = Math.Sqrt(Math.Pow(position.X - predX.Value, 2) + 
                                   Math.Pow(position.Y - predY.Value, 2) + 
                                   Math.Pow(position.Z - predZ.Value, 2));
            score -= distance * 0.1;
        }
        
        // 6. Kutu boyutlarına göre bonus (daha büyük kutular öncelikli)
        score += (box.Width * box.Height * box.Depth) * 0.001;
        
        // 7. Kenar hizalama bonusu - konteyner kenarlarına yakın olma
        var edgeAlignment = CalculateEdgeAlignment(position, box, container);
        score += edgeAlignment * 20;
        
        // 8. Diğer kutularla temas bonusu
        var contactBonus = CalculateContactBonus(position, box, container);
        score += contactBonus * 10;
        
        return score;
    }

    static double CalculateSpaceEfficiency(Position position, Box box, Container container)
    {
        var efficiency = 0.0;
        
        // Kutu yerleştirildikten sonra oluşacak boşlukları hesapla
        var rightSpace = container.Width - (position.X + box.Width);
        var topSpace = container.Height - (position.Y + box.Height);
        var frontSpace = container.Depth - (position.Z + box.Depth);
        
        // Küçük boşluklar daha verimli (daha az israf)
        if (rightSpace > 0 && rightSpace < 10) efficiency += 1.0;
        if (topSpace > 0 && topSpace < 10) efficiency += 1.0;
        if (frontSpace > 0 && frontSpace < 10) efficiency += 1.0;
        
        // Çok büyük boşluklar verimsiz
        if (rightSpace > 20) efficiency -= 0.5;
        if (topSpace > 20) efficiency -= 0.5;
        if (frontSpace > 20) efficiency -= 0.5;
        
        return efficiency;
    }

    static double CalculateEdgeAlignment(Position position, Box box, Container container)
    {
        var alignment = 0.0;
        
        // Sol kenara yakınlık
        if (position.X <= 5) alignment += 1.0;
        else if (position.X <= 10) alignment += 0.5;
        
        // Alt kenara yakınlık
        if (position.Y <= 5) alignment += 1.0;
        else if (position.Y <= 10) alignment += 0.5;
        
        // Arka kenara yakınlık
        if (position.Z <= 5) alignment += 1.0;
        else if (position.Z <= 10) alignment += 0.5;
        
        // Sağ kenara tam oturma
        if (position.X + box.Width >= container.Width - 5) alignment += 0.5;
        
        // Üst kenara tam oturma
        if (position.Y + box.Height >= container.Height - 5) alignment += 0.5;
        
        // Ön kenara tam oturma
        if (position.Z + box.Depth >= container.Depth - 5) alignment += 0.5;
        
        return alignment;
    }

    static double CalculateContactBonus(Position position, Box box, Container container)
    {
        var contact = 0.0;
        
        // Diğer kutularla temas eden yüzey sayısını hesapla
        // Bu basit bir yaklaşım - gerçek implementasyonda daha karmaşık olabilir
        
        // Sol yüzey teması
        if (position.X == 0) contact += 0.5;
        
        // Alt yüzey teması
        if (position.Y == 0) contact += 0.5;
        
        // Arka yüzey teması
        if (position.Z == 0) contact += 0.5;
        
        return contact;
    }

    static Position? FindOptimalPosition(Container container, Box box)
    {
        // En düşük Z koordinatından başla
        for (int z = 0; z <= container.Depth - box.Depth; z++)
        {
            for (int y = 0; y <= container.Height - box.Height; y++)
            {
                for (int x = 0; x <= container.Width - box.Width; x++)
                {
                    if (container.CanPlaceBox(x, y, z, box.Width, box.Height, box.Depth))
                    {
                        return new Position(x, y, z);
                    }
                }
            }
        }
        
        return null;
    }

    static Position? FindBestPosition(Container container, Box box, int predX, int predY, int predZ)
    {
        // Tahmin edilen pozisyona yakın pozisyonları dene
        var searchRadius = 10;
        
        for (int dx = -searchRadius; dx <= searchRadius; dx++)
        {
            for (int dy = -searchRadius; dy <= searchRadius; dy++)
            {
                for (int dz = -searchRadius; dz <= searchRadius; dz++)
                {
                    var x = predX + dx;
                    var y = predY + dy;
                    var z = predZ + dz;
                    
                    if (x >= 0 && y >= 0 && z >= 0 &&
                        x + box.Width <= container.Width &&
                        y + box.Height <= container.Height &&
                        z + box.Depth <= container.Depth &&
                        container.CanPlaceBox(x, y, z, box.Width, box.Height, box.Depth))
                    {
                        return new Position(x, y, z);
                    }
                }
            }
        }
        
        // Eğer tahmin edilen pozisyon yakınında yer bulunamazsa, tüm konteyneri tara
        for (int x = 0; x <= container.Width - box.Width; x++)
        {
            for (int y = 0; y <= container.Height - box.Height; y++)
            {
                for (int z = 0; z <= container.Depth - box.Depth; z++)
                {
                    if (container.CanPlaceBox(x, y, z, box.Width, box.Height, box.Depth))
                    {
                        return new Position(x, y, z);
                    }
                }
            }
        }
        
        return null;
    }
}

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
    
    public Container(int w, int h, int d)
    {
        Width = w; Height = h; Depth = d;
        occupied = new bool[w, h, d];
    }
    
    public bool CanPlaceBox(int x, int y, int z, int w, int h, int d)
    {
        for (int i = x; i < x + w; i++)
            for (int j = y; j < y + h; j++)
                for (int k = z; k < z + d; k++)
                    if (occupied[i, j, k]) return false;
        return true;
    }
    
    public void PlaceBox(int x, int y, int z, int w, int h, int d)
    {
        for (int i = x; i < x + w; i++)
            for (int j = y; j < y + h; j++)
                for (int k = z; k < z + d; k++)
                    occupied[i, j, k] = true;
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

