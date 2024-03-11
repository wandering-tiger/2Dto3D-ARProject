using UnityEngine;
using System.Collections.Generic;
using System.Globalization;

public class SkinningWithVoxelHeat : MonoBehaviour
{
    public SkinnedMeshRenderer skinnedMeshRenderer;
    public Transform[] bones;
    public VoxelHeatDiffusion voxelHeatDiffusion;

    private void Start()
    {
        // 执行体素热扩散计算
        voxelHeatDiffusion.DiffuseHeat();

        // 根据体素热量结果设置骨骼权重
        SetBoneWeights();
    }

    private void SetBoneWeights()
    {
        Mesh mesh = skinnedMeshRenderer.sharedMesh;
        BoneWeight[] boneWeights = new BoneWeight[mesh.vertexCount];

        // 遍历网格的顶点
        for (int i = 0; i < mesh.vertexCount; i++)
        {
            Vector3 vertex = mesh.vertices[i];

            // 将顶点坐标从世界空间转换为局部空间
            Vector3 localVertex = skinnedMeshRenderer.transform.InverseTransformPoint(transform.TransformPoint(vertex));

            // 获取顶点所在体素的坐标
            int x = Mathf.FloorToInt((localVertex.x + 0.5f) * voxelHeatDiffusion.gridSize);
            int y = Mathf.FloorToInt((localVertex.y + 0.5f) * voxelHeatDiffusion.gridSize);
            int z = Mathf.FloorToInt((localVertex.z + 0.5f) * voxelHeatDiffusion.gridSize);

            // 获取顶点所在体素的热量信息
            List<HeatInfo> heatInfos = GetHeatInfo(x, y, z);

            // 计算骨骼权重
            boneWeights[i] = CalculateBoneWeight(heatInfos);
        }

        // 将新的骨骼权重赋给网格
        mesh.boneWeights = boneWeights;

        // 更新蒙皮网格
        skinnedMeshRenderer.sharedMesh = mesh;
    }

    private List<HeatInfo> GetHeatInfo(int x, int y, int z)
    {
        List<HeatInfo> heatInfos = new List<HeatInfo>();

        // 遍历所有骨骼，获取热量值
        for (int i = 0; i < bones.Length; i++)
        {
            float heat = voxelHeatDiffusion.GetHeatValue(x, y, z, i);
            heatInfos.Add(new HeatInfo(i, heat));
        }

        // 根据热量值进行排序
        heatInfos.Sort((a, b) => b.heat.CompareTo(a.heat));

        // 只保留热量值最高的前四个
        heatInfos = heatInfos.GetRange(0, Mathf.Min(4, heatInfos.Count));

        // 对权重进行归一化处理
        float totalHeat = 0f;
        foreach (HeatInfo heatInfo in heatInfos)
        {
            totalHeat += heatInfo.heat;
        }

        foreach (HeatInfo heatInfo in heatInfos)
        {
            heatInfo.heat /= totalHeat;
        }
        return heatInfos;
    }

    private BoneWeight CalculateBoneWeight(List<HeatInfo> heatInfos)
    {
        // 构建骨骼权重
        BoneWeight boneWeight = new BoneWeight();

        // 设置骨骼权重
        for (int i = 0; i < heatInfos.Count; i++)
        {
            switch (i)
            {
                case 0:
                    boneWeight.boneIndex0 = heatInfos[i].boneIndex;
                    boneWeight.weight0 = heatInfos[i].heat;
                    break;
                case 1:
                    boneWeight.boneIndex1 = heatInfos[i].boneIndex;
                    boneWeight.weight1 = heatInfos[i].heat;
                    break;
                case 2:
                    boneWeight.boneIndex2 = heatInfos[i].boneIndex;
                    boneWeight.weight2 = heatInfos[i].heat;
                    break;
                case 3:
                    boneWeight.boneIndex3 = heatInfos[i].boneIndex;
                    boneWeight.weight3 = heatInfos[i].heat;
                    break;
            }
        }

        return boneWeight;
    }

    private class HeatInfo
    {
        public int boneIndex;
        public float heat;

        public HeatInfo(int boneIndex, float heat)
        {
            this.boneIndex = boneIndex;
            this.heat = heat;
        }
    }
}


