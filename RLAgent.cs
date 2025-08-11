using System;
using System.Collections.Generic;
using System.Linq;
// BoxTypes.cs'deki sınıfları kullan
// using BoxTypes;

// namespace BoxML
    public class RLAction
    {
        public Position Position { get; set; }
        public string Rotation { get; set; }
        
        public RLAction(Position position, string rotation)
        {
            Position = position;
            Rotation = rotation;
        }
        
        public override int GetHashCode()
        {
            return HashCode.Combine(Position.X, Position.Y, Position.Z, Rotation);
        }
    }

    public class RLAgent
    {
        private Dictionary<string, double> qTable = new();
        private Random random = new();
        public double TotalReward { get; private set; } = 0;
        public int ActionsTaken { get; private set; } = 0;
        public double SuccessRate { get; private set; } = 0;
        public double LearningScore { get; private set; } = 0;
        public int SuccessfulPlacements { get; private set; } = 0;
        private int episodeCount = 0;
        
        public RLAction GetAction(RLState state, Box box)
        {
            var rotations = new[] { "XYZ", "XZY", "YXZ", "ZYX", "ZXY", "YZX" };
            var bestAction = (RLAction?)null;
            var bestQValue = double.MinValue;
            
            // Epsilon-greedy strateji (keşif vs sömürü) - daha yüksek exploration
            var epsilon = Math.Max(0.05, 0.3 - (episodeCount * 0.0001)); // Başlangıçta %30, zamanla azalır
            if (random.NextDouble() < epsilon)
            {
                // Keşif: rastgele aksiyon
                var randomRotation = rotations[random.Next(rotations.Length)];
                var randomPosition = new Position(
                    random.Next(0, Math.Max(1, 100 - box.Width)),
                    random.Next(0, Math.Max(1, 100 - box.Height)),
                    random.Next(0, Math.Max(1, 100 - box.Depth))
                );
                return new RLAction(randomPosition, randomRotation);
            }
            
            // Sömürü: en iyi aksiyonu seç - daha ince grid
            foreach (var rotation in rotations)
            {
                for (int x = 0; x <= 100 - box.Width; x += 2) // 5'ten 2'ye düşürdük
                {
                    for (int y = 0; y <= 100 - box.Height; y += 2)
                    {
                        for (int z = 0; z <= 100 - box.Depth; z += 2)
                        {
                            var action = new RLAction(new Position(x, y, z), rotation);
                            var stateActionKey = $"{state.GetHashCode()}_{action.GetHashCode()}";
                            
                            var qValue = qTable.GetValueOrDefault(stateActionKey, 0.0);
                            if (qValue > bestQValue)
                            {
                                bestQValue = qValue;
                                bestAction = action;
                            }
                        }
                    }
                }
            }
            
            return bestAction ?? new RLAction(new Position(0, 0, 0), "XYZ");
        }
        
        public void Update(RLAction action, double reward, RLState newState)
        {
            TotalReward += reward;
            ActionsTaken++;
            
            if (reward > 0) SuccessfulPlacements++;
            SuccessRate = (double)SuccessfulPlacements / ActionsTaken;
            
            // Q-learning güncelleme - daha yüksek learning rate
            var learningRate = 0.2; // 0.1'den 0.2'ye çıkardık
            var discountFactor = 0.95; // 0.9'dan 0.95'e çıkardık
            
            // Basit Q-learning implementasyonu
            var stateActionKey = $"{newState.GetHashCode()}_{action.GetHashCode()}";
            var currentQValue = qTable.GetValueOrDefault(stateActionKey, 0.0);
            var newQValue = currentQValue + learningRate * (reward + discountFactor * GetMaxQValue(newState) - currentQValue);
            qTable[stateActionKey] = newQValue;
            
            // Öğrenme skorunu güncelle
            LearningScore = TotalReward / Math.Max(1, ActionsTaken);
        }
        
        public void StartEpisode()
        {
            episodeCount++;
        }
        
        private double GetMaxQValue(RLState state)
        {
            var maxQValue = 0.0;
            foreach (var kvp in qTable)
            {
                if (kvp.Key.StartsWith(state.GetHashCode().ToString()))
                {
                    maxQValue = Math.Max(maxQValue, kvp.Value);
                }
            }
            return maxQValue;
        }
        
        public void SaveModel(string filename)
        {
            // Q-table'ı JSON olarak kaydet
            var json = System.Text.Json.JsonSerializer.Serialize(qTable, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filename, json);
        }
        
        public void LoadModel(string filename)
        {
            if (File.Exists(filename))
            {
                var json = File.ReadAllText(filename);
                qTable = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, double>>(json) ?? new();
            }
        }
    }

    public class RLState
    {
        public double FillRate { get; set; }
        public int PlacedBoxes { get; set; }
        public double AverageBoxSize { get; set; }
        public double ContainerHeight { get; set; }
        public double ContainerWidth { get; set; }
        public double ContainerDepth { get; set; }
        public double RemainingVolume { get; set; }
        
        public RLState(double fillRate, int placedBoxes, double averageBoxSize, 
                      double containerHeight, double containerWidth, double containerDepth, double remainingVolume)
        {
            FillRate = fillRate;
            PlacedBoxes = placedBoxes;
            AverageBoxSize = averageBoxSize;
            ContainerHeight = containerHeight;
            ContainerWidth = containerWidth;
            ContainerDepth = containerDepth;
            RemainingVolume = remainingVolume;
        }
        
        public override int GetHashCode()
        {
            return HashCode.Combine(
                Math.Round(FillRate, 1),
                PlacedBoxes,
                Math.Round(AverageBoxSize, 1),
                Math.Round(ContainerHeight, 1),
                Math.Round(ContainerWidth, 1),
                Math.Round(ContainerDepth, 1),
                Math.Round(RemainingVolume, 1)
            );
        }
    }

    public class RLEnvironment
    {
        private Container container;
        private List<PlacedBox> placedBoxes;
        
        public RLEnvironment(Container container)
        {
            this.container = container;
            this.placedBoxes = new List<PlacedBox>();
        }
        
        public RLState GetState()
        {
            var totalVolume = placedBoxes.Sum(b => b.Volume);
            var fillRate = (totalVolume / (container.Width * container.Height * container.Depth)) * 100;
            var averageBoxSize = placedBoxes.Count > 0 ? placedBoxes.Average(b => b.Volume) : 0;
            var remainingVolume = (container.Width * container.Height * container.Depth) - totalVolume;
            
            return new RLState(
                fillRate,
                placedBoxes.Count,
                averageBoxSize,
                container.Height,
                container.Width,
                container.Depth,
                remainingVolume
            );
        }
        
        public void Update(Container container, List<PlacedBox> placedBoxes)
        {
            this.container = container;
            this.placedBoxes = new List<PlacedBox>(placedBoxes);
        }
    }

    public static class RLHelper
    {
        public static Box ApplyRotation(Box originalBox, string rotation)
        {
            return rotation switch
            {
                "XYZ" => originalBox,
                "XZY" => new Box(originalBox.Width, originalBox.Depth, originalBox.Height, originalBox.Weight),
                "YXZ" => new Box(originalBox.Height, originalBox.Width, originalBox.Depth, originalBox.Weight),
                "ZYX" => new Box(originalBox.Depth, originalBox.Height, originalBox.Width, originalBox.Weight),
                "ZXY" => new Box(originalBox.Depth, originalBox.Width, originalBox.Height, originalBox.Weight),
                "YZX" => new Box(originalBox.Height, originalBox.Depth, originalBox.Width, originalBox.Weight),
                _ => originalBox
            };
        }

        public static Position? CalculatePosition(Container container, Box box, RLAction action)
        {
            // Aksiyona göre pozisyon hesapla
            var x = Math.Max(0, Math.Min(container.Width - box.Width, action.Position.X));
            var y = Math.Max(0, Math.Min(container.Height - box.Height, action.Position.Y));
            var z = Math.Max(0, Math.Min(container.Depth - box.Depth, action.Position.Z));
            
            return new Position(x, y, z);
        }

        public static double CalculateReward(PlacementResult placement, Container container, double totalVolume, int placedCount)
        {
            var reward = 0.0;
            
            // Başarılı yerleştirme bonusu - daha yüksek
            reward += 2000;
            
            // Doluluk oranı bonusu - daha agresif
            var fillRate = (totalVolume / (100 * 100 * 100)) * 100;
            if (fillRate > 95) reward += 1000;
            else if (fillRate > 90) reward += 800;
            else if (fillRate > 85) reward += 600;
            else if (fillRate > 80) reward += 400;
            else if (fillRate > 75) reward += 200;
            else if (fillRate > 70) reward += 100;
            
            // Kutu sayısı bonusu
            reward += placedCount * 50;
            
            // Kenar hizalama bonusu - daha yüksek
            if (placement.Position.X <= 2 || placement.Position.X + placement.Box.Width >= container.Width - 2) reward += 300;
            if (placement.Position.Y <= 2 || placement.Position.Y + placement.Box.Height >= container.Height - 2) reward += 300;
            if (placement.Position.Z <= 2 || placement.Position.Z + placement.Box.Depth >= container.Depth - 2) reward += 300;
            
            // Düşük koordinat bonusu (yerçekimi) - daha az ceza
            reward -= placement.Position.Z * 5; // 10'dan 5'e düşürdük
            reward -= placement.Position.Y * 2; // 5'ten 2'ye düşürdük
            reward -= placement.Position.X * 1; // 2'den 1'e düşürdük
            
            // Boşluk analizi - daha iyi
            var rightSpace = container.Width - (placement.Position.X + placement.Box.Width);
            var topSpace = container.Height - (placement.Position.Y + placement.Box.Height);
            var frontSpace = container.Depth - (placement.Position.Z + placement.Box.Depth);
            
            if (rightSpace > 0 && rightSpace < 5) reward += 200; // 10'dan 5'e düşürdük
            if (topSpace > 0 && topSpace < 5) reward += 200;
            if (frontSpace > 0 && frontSpace < 5) reward += 200;
            
            // Kompaktlık bonusu
            if (placement.Position.X == 0 || placement.Position.Y == 0 || placement.Position.Z == 0) reward += 150;
            
            return reward;
        }

        public static PlacementResult? ExecuteAction(Container container, Box originalBox, RLAction action)
        {
            // Aksiyona göre kutuyu döndür
            var rotatedBox = ApplyRotation(originalBox, action.Rotation);
            
            // Aksiyona göre pozisyonu hesapla
            var position = CalculatePosition(container, rotatedBox, action);
            
            if (position != null && container.CanPlaceBox(position.X, position.Y, position.Z, rotatedBox.Width, rotatedBox.Height, rotatedBox.Depth))
            {
                return new PlacementResult
                {
                    Position = position,
                    Box = rotatedBox,
                    Rotation = action.Rotation
                };
            }
            
            return null;
        }
    } 