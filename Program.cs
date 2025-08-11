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

        // Kutu yerleştirme algoritması (gelişmiş optimizasyon ile)
        var container = new Container(100, 100, 100);
        var placedBoxes = new List<PlacedBox>();
        var totalVolume = 0.0;
        var placedCount = 0;

        // Kutuları akıllı sırala (hacim + şekil faktörü + kompaktlık)
        var sortedBoxes = dataList
            .OrderByDescending(b => b.Volume) // Önce büyük kutular
            .ThenByDescending(b => Math.Max(b.X, Math.Max(b.Y, b.Z))) // En uzun kenar
            .ThenByDescending(b => b.Weight) // Ağır kutular önce
            .ThenByDescending(b => CalculateCompactness((int)b.X, (int)b.Y, (int)b.Z)) // Kompaktlık
            .ToList();

        Console.WriteLine("Kutu yerleştirme başlıyor (tam optimizasyon algoritması ile)...");
        Console.WriteLine($"Toplam {sortedBoxes.Count} kutu işlenecek\n");

        // İlk geçiş: Büyük kutuları yerleştir
        var largeBoxes = sortedBoxes.Where(b => b.Volume > 200).ToList();
        var smallBoxes = sortedBoxes.Where(b => b.Volume <= 200).ToList();

        Console.WriteLine($"İlk geçiş: {largeBoxes.Count} büyük kutu yerleştirilecek");
        int margin = 1; // Marjinli yerleştirme için
        foreach (var boxData in largeBoxes)
        {
            var box = new Box((int)boxData.X, (int)boxData.Y, (int)boxData.Z, boxData.Weight);
            var placement = FindOptimalPositionWithRotationAdvanced(container, box);
            if (placement != null && container.CanPlaceBox(placement.Position.X, placement.Position.Y, placement.Position.Z, placement.Box.Width, placement.Box.Height, placement.Box.Depth, margin))
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
                container.PlaceBox(placement.Position.X, placement.Position.Y, placement.Position.Z, placement.Box.Width, placement.Box.Height, placement.Box.Depth, margin);
                var originalVolume = boxData.X * boxData.Y * boxData.Z;
                var efficiency = (placedBox.Volume / originalVolume) * 100;
                Console.WriteLine($"Büyük Kutu {placedCount}: {boxData.X}x{boxData.Y}x{boxData.Z} -> {placement.Box.Width}x{placement.Box.Height}x{placement.Box.Depth} (Rotasyon: {placement.Rotation}) | Verimlilik: {efficiency:F1}% | Pozisyon: ({placement.Position.X},{placement.Position.Y},{placement.Position.Z})");
            }
            else
            {
                Console.WriteLine($"❌ Büyük kutu yerleştirilemedi: {boxData.X}x{boxData.Y}x{boxData.Z} (Hacim: {boxData.Volume})");
            }
        }

        Console.WriteLine($"\nİkinci geçiş: {smallBoxes.Count} küçük kutu yerleştirilecek");
        foreach (var boxData in smallBoxes)
        {
            var box = new Box((int)boxData.X, (int)boxData.Y, (int)boxData.Z, boxData.Weight);
            var placement = FindOptimalPositionWithRotationForSmallBoxes(container, box, gridStep: 1);
            if (placement != null && container.CanPlaceBox(placement.Position.X, placement.Position.Y, placement.Position.Z, placement.Box.Width, placement.Box.Height, placement.Box.Depth, margin))
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
                container.PlaceBox(placement.Position.X, placement.Position.Y, placement.Position.Z, placement.Box.Width, placement.Box.Height, placement.Box.Depth, margin);
                var originalVolume = boxData.X * boxData.Y * boxData.Z;
                var efficiency = (placedBox.Volume / originalVolume) * 100;
                Console.WriteLine($"Küçük Kutu {placedCount}: {boxData.X}x{boxData.Y}x{boxData.Z} -> {placement.Box.Width}x{placement.Box.Height}x{placement.Box.Depth} (Rotasyon: {placement.Rotation}) | Verimlilik: {efficiency:F1}% | Pozisyon: ({placement.Position.X},{placement.Position.Y},{placement.Position.Z})");
            }
            else
            {
                Console.WriteLine($"❌ Küçük kutu yerleştirilemedi: {boxData.X}x{boxData.Y}x{boxData.Z} (Hacim: {boxData.Volume})");
            }
        }

        // 3. Geçiş: En küçük kutuları kalan minik boşluklara sıkıştır
        var tinyBoxes = smallBoxes.Where(b => b.Volume <= 300).OrderBy(b => b.Volume).ToList();
        Console.WriteLine($"\nÜçüncü geçiş: {tinyBoxes.Count} minik kutu yerleştirilecek");
        foreach (var boxData in tinyBoxes)
        {
            var box = new Box((int)boxData.X, (int)boxData.Y, (int)boxData.Z, boxData.Weight);
            var placement = FindOptimalPositionWithRotationForSmallBoxes(container, box, gridStep: 1, tightFit: true);
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
                container.PlaceBox(placement.Position.X, placement.Position.Y, placement.Position.Z, 
                                 placement.Box.Width, placement.Box.Height, placement.Box.Depth);
                Console.WriteLine($"Minik Kutu {placedCount}: {boxData.X}x{boxData.Y}x{boxData.Z} -> {placement.Box.Width}x{placement.Box.Height}x{placement.Box.Depth} (Rotasyon: {placement.Rotation}) | Pozisyon: ({placement.Position.X},{placement.Position.Y},{placement.Position.Z})");
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

    static double CalculateCompactness(int x, int y, int z)
    {
        // Küp şeklindeki kutular daha kompakt
        var max = Math.Max(x, Math.Max(y, z));
        var min = Math.Min(x, Math.Min(y, z));
        return (double)min / max; // 1'e yakın = daha kompakt
    }

    static PlacementResult? FindOptimalPositionWithRotationAdvanced(Container container, Box originalBox)
    {
        var bestPlacement = (PlacementResult?)null;
        var bestScore = double.MinValue;

        // Tüm olası rotasyonları dene
        var rotations = GetBoxRotations(originalBox);
        
        foreach (var rotatedBox in rotations)
        {
            var position = FindOptimalPositionAdvanced(container, rotatedBox.Box);
            if (position != null)
            {
                // Gelişmiş skor hesapla
                var score = CalculateAdvancedPlacementScore(position, rotatedBox.Box, container);
                
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

    static PlacementResult? FindOptimalPositionWithRotationForSmallBoxes(Container container, Box originalBox, int gridStep = 1, bool tightFit = false)
    {
        var bestPlacement = (PlacementResult?)null;
        var bestScore = double.MinValue;

        // Tüm olası rotasyonları dene
        var rotations = GetBoxRotations(originalBox);
        
        foreach (var rotatedBox in rotations)
        {
            var position = FindOptimalPositionForSmallBoxes(container, rotatedBox.Box, gridStep, tightFit);
            if (position != null)
            {
                // Küçük kutular için özel skor hesapla
                var score = CalculateSmallBoxPlacementScore(position, rotatedBox.Box, container, tightFit);
                
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

    static double CalculateAdvancedPlacementScore(Position position, Box box, Container container)
    {
        var score = 0.0;
        
        // 1. Yerçekimi etkisi - daha düşük Z koordinatı tercih edilir
        score -= position.Z * 20;
        
        // 2. Alt kısım tercihi - daha düşük Y koordinatı
        score -= position.Y * 10;
        
        // 3. Sol taraf tercihi - daha düşük X koordinatı
        score -= position.X * 5;
        
        // 4. Gelişmiş boşluk analizi
        var spaceEfficiency = CalculateAdvancedSpaceEfficiency(position, box, container);
        score += spaceEfficiency * 100;
        
        // 5. Kutu boyutlarına göre bonus (daha büyük kutular öncelikli)
        score += (box.Width * box.Height * box.Depth) * 0.002;
        
        // 6. Gelişmiş kenar hizalama
        var edgeAlignment = CalculateAdvancedEdgeAlignment(position, box, container);
        score += edgeAlignment * 50;
        
        // 7. Gelişmiş temas bonusu
        var contactBonus = CalculateAdvancedContactBonus(position, box, container);
        score += contactBonus * 30;
        
        // 8. Kompaktlık bonusu
        var compactness = CalculateCompactness(box.Width, box.Height, box.Depth);
        score += compactness * 200;
        
        return score;
    }

    static double CalculateSmallBoxPlacementScore(Position position, Box box, Container container, bool tightFit = false)
    {
        var score = 0.0;
        
        // Küçük kutular için farklı strateji
        // 1. Boşluk doldurma önceliği
        var spaceEfficiency = CalculateSmallBoxSpaceEfficiency(position, box, container);
        score += spaceEfficiency * 200;
        
        // 2. Daha az yerçekimi etkisi
        score -= position.Z * 10;
        score -= position.Y * 5;
        score -= position.X * 2;
        
        // 3. Kenar teması daha önemli
        var contactBonus = CalculateAdvancedContactBonus(position, box, container);
        score += contactBonus * 80;
        
        // 4. Küçük boşluklara sıkışma bonusu
        var tightFitBonus = CalculateTightFitBonus(position, box, container);
        score += tightFitBonus * 100;
        
        if (tightFit) score += 500; // Sıkışma bonusu
        
        return score;
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

    static double CalculateAdvancedSpaceEfficiency(Position position, Box box, Container container)
    {
        var efficiency = 0.0;
        
        // Kutu yerleştirildikten sonra oluşacak boşlukları hesapla
        var rightSpace = container.Width - (position.X + box.Width);
        var topSpace = container.Height - (position.Y + box.Height);
        var frontSpace = container.Depth - (position.Z + box.Depth);
        
        // Küçük boşluklar daha verimli (daha az israf)
        if (rightSpace > 0 && rightSpace < 5) efficiency += 2.0;
        else if (rightSpace > 0 && rightSpace < 10) efficiency += 1.0;
        if (topSpace > 0 && topSpace < 5) efficiency += 2.0;
        else if (topSpace > 0 && topSpace < 10) efficiency += 1.0;
        if (frontSpace > 0 && frontSpace < 5) efficiency += 2.0;
        else if (frontSpace > 0 && frontSpace < 10) efficiency += 1.0;
        
        // Çok büyük boşluklar verimsiz
        if (rightSpace > 15) efficiency -= 1.0;
        if (topSpace > 15) efficiency -= 1.0;
        if (frontSpace > 15) efficiency -= 1.0;
        
        return efficiency;
    }

    static double CalculateSmallBoxSpaceEfficiency(Position position, Box box, Container container)
    {
        var efficiency = 0.0;
        
        // Küçük kutular için daha agresif boşluk analizi
        var rightSpace = container.Width - (position.X + box.Width);
        var topSpace = container.Height - (position.Y + box.Height);
        var frontSpace = container.Depth - (position.Z + box.Depth);
        
        // Çok küçük boşluklar ideal
        if (rightSpace > 0 && rightSpace < 3) efficiency += 3.0;
        else if (rightSpace > 0 && rightSpace < 5) efficiency += 2.0;
        if (topSpace > 0 && topSpace < 3) efficiency += 3.0;
        else if (topSpace > 0 && topSpace < 5) efficiency += 2.0;
        if (frontSpace > 0 && frontSpace < 3) efficiency += 3.0;
        else if (frontSpace > 0 && frontSpace < 5) efficiency += 2.0;
        
        return efficiency;
    }

    static double CalculateTightFitBonus(Position position, Box box, Container container)
    {
        var bonus = 0.0;
        
        // Diğer kutularla temas eden yüzey sayısı
        var contactSurfaces = 0;
        
        // Sol yüzey teması
        if (position.X == 0) contactSurfaces++;
        
        // Alt yüzey teması
        if (position.Y == 0) contactSurfaces++;
        
        // Arka yüzey teması
        if (position.Z == 0) contactSurfaces++;
        
        // Sağ yüzey teması
        if (position.X + box.Width >= container.Width - 1) contactSurfaces++;
        
        // Üst yüzey teması
        if (position.Y + box.Height >= container.Height - 1) contactSurfaces++;
        
        // Ön yüzey teması
        if (position.Z + box.Depth >= container.Depth - 1) contactSurfaces++;
        
        bonus = contactSurfaces * 50;
        
        return bonus;
    }

    static double CalculateAdvancedEdgeAlignment(Position position, Box box, Container container)
    {
        var alignment = 0.0;
        
        // Sol kenara yakınlık
        if (position.X <= 2) alignment += 2.0;
        else if (position.X <= 5) alignment += 1.0;
        else if (position.X <= 10) alignment += 0.5;
        
        // Alt kenara yakınlık
        if (position.Y <= 2) alignment += 2.0;
        else if (position.Y <= 5) alignment += 1.0;
        else if (position.Y <= 10) alignment += 0.5;
        
        // Arka kenara yakınlık
        if (position.Z <= 2) alignment += 2.0;
        else if (position.Z <= 5) alignment += 1.0;
        else if (position.Z <= 10) alignment += 0.5;
        
        // Sağ kenara tam oturma
        if (position.X + box.Width >= container.Width - 2) alignment += 1.0;
        else if (position.X + box.Width >= container.Width - 5) alignment += 0.5;
        
        // Üst kenara tam oturma
        if (position.Y + box.Height >= container.Height - 2) alignment += 1.0;
        else if (position.Y + box.Height >= container.Height - 5) alignment += 0.5;
        
        // Ön kenara tam oturma
        if (position.Z + box.Depth >= container.Depth - 2) alignment += 1.0;
        else if (position.Z + box.Depth >= container.Depth - 5) alignment += 0.5;
        
        return alignment;
    }

    static double CalculateAdvancedContactBonus(Position position, Box box, Container container)
    {
        var contact = 0.0;
        
        // Diğer kutularla temas eden yüzey sayısını hesapla
        // Bu basit bir yaklaşım - gerçek implementasyonda daha karmaşık olabilir
        
        // Sol yüzey teması
        if (position.X == 0) contact += 1.0;
        
        // Alt yüzey teması
        if (position.Y == 0) contact += 1.0;
        
        // Arka yüzey teması
        if (position.Z == 0) contact += 1.0;
        
        // Sağ yüzey teması
        if (position.X + box.Width >= container.Width - 1) contact += 0.5;
        
        // Üst yüzey teması
        if (position.Y + box.Height >= container.Height - 1) contact += 0.5;
        
        // Ön yüzey teması
        if (position.Z + box.Depth >= container.Depth - 1) contact += 0.5;
        
        return contact;
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

    static Position? FindOptimalPositionAdvanced(Container container, Box box)
    {
        // Dinamik grid - kutu boyutuna göre grid kalınlığı
        var gridStep = Math.Max(1, Math.Min(3, Math.Min(box.Width, Math.Min(box.Height, box.Depth)) / 5));
        
        // En düşük Z koordinatından başla
        for (int z = 0; z <= container.Depth - box.Depth; z += gridStep)
        {
            for (int y = 0; y <= container.Height - box.Height; y += gridStep)
            {
                for (int x = 0; x <= container.Width - box.Width; x += gridStep)
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

    static Position? FindOptimalPositionForSmallBoxes(Container container, Box box, int gridStep = 1, bool tightFit = false)
    {
        for (int z = 0; z <= container.Depth - box.Depth; z += gridStep)
        {
            for (int y = 0; y <= container.Height - box.Height; y += gridStep)
            {
                for (int x = 0; x <= container.Width - box.Width; x += gridStep)
                {
                    if (container.CanPlaceBox(x, y, z, box.Width, box.Height, box.Depth))
                    {
                        if (tightFit)
                        {
                            // Sadece tam sıkışan pozisyonları al
                            if (IsTightFit(container, x, y, z, box.Width, box.Height, box.Depth))
                                return new Position(x, y, z);
                        }
                        else
                        {
                            return new Position(x, y, z);
                        }
                    }
                }
            }
        }
        return null;
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

    static bool IsTightFit(Container container, int x, int y, int z, int w, int h, int d)
    {
        // Kutunun etrafında başka kutu veya konteyner duvarı olmalı
        bool tight = false;
        if (x == 0 || x + w == container.Width) tight = true;
        if (y == 0 || y + h == container.Height) tight = true;
        if (z == 0 || z + d == container.Depth) tight = true;
        // Yanında başka kutu var mı (basit kontrol)
        // (Daha gelişmiş temas kontrolü eklenebilir)
        return tight;
    }
} 