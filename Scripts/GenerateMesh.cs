using System.Collections.Generic;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using TriangleNet.Geometry;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TriangleNet.Meshing;
using Utils = OpenCVForUnity.UnityUtils.Utils;
using System.Linq;
using UnityEditor;
using Vertex = TriangleNet.Geometry.Vertex;
using System;
using Random = UnityEngine.Random;
using Rect = UnityEngine.Rect;
using TriangleNet.Meshing.Algorithm;
using OpenCVForUnity.PhotoModule;
using Unity.VisualScripting;
using System.Security.Cryptography;
using UnityEngine.Networking;
using System.Text;
//using Autodesk.Fbx;
using System.IO;
using TensorFlow;
using OpenCvSharp;

//using UnityEngine.AI;

public class GenerateMesh : MonoBehaviour
{
    // Start is called before the first frame update
    public RawImage Pic;
    //轮廓点
    public List<OpenCVForUnity.CoreModule.Point> POINTS;
    //数组形式的轮廓点
    List<Vector3> bounpoints = new List<Vector3>();
    //网格间距
    public float gridSpacing;
    //轮廓线
    public Polygon polygon = new Polygon();
    private GameObject frontFace;
    private GameObject backFace;
    public Texture2D texture;
    public Texture2D texture1;
    public Material[] mats;
    public string materialName = "Cloud"; // Material的名称
    public Mesh meshF;
    public double HP;


    
    private void Start()
    {
        string p= Application.dataPath + "/Resources/Cloud.png";
        //string p = "D:\\Unity\\project\\test\\Assets\\Resources\\Elephant.png";
        SetMeshAndMats(p);
        //MeshToString();
    }

    public void SetMeshAndMats(string path)
    {
        buildChildMesh();
        string imagePath = path;
        makeMesh(imagePath);
    }

    void buildChildMesh()
    {
        frontFace = new GameObject();
        backFace = new GameObject();
        //GameObject frontIns = Instantiate(frontFace);
        frontFace.name = "Front";
        frontFace.transform.parent = gameObject.transform;
        frontFace.AddComponent<MeshFilter>();
        frontFace.AddComponent<MeshRenderer>();
        frontFace.AddComponent<MeshCollider>();
        //GameObject backIns = Instantiate(backFace);
        backFace.name = "Back";
        backFace.transform.parent = gameObject.transform;
        backFace.AddComponent<MeshFilter>();
        backFace.AddComponent<MeshRenderer>();
        backFace.AddComponent<MeshCollider>();
    }

    private void destroyChildMesh()
    {
        Destroy(frontFace);
        Destroy(backFace);
    }

    void makeMesh(string path)
    {
        findContours(path);
        makeSubMesh(path);
        makeFinalMesh();
    }

    void findContours(string path)
    {

        //读取图片并设置尺寸
        Mat mat=Imgcodecs.imread(path,-1);
        Mat mat1 = new Mat();
        int w = mat.cols();
        int h = mat.rows();
        HP = 400 * h / w;
        Imgproc.resize(mat, mat, new Size(400, 400 * h / w));
        Imgproc.GaussianBlur(mat, mat, new Size(400, 400 * h / w), 0);

        //转为灰度图
        Imgproc.cvtColor(mat, mat1, Imgproc.COLOR_BGR2GRAY);
        Core.flip(mat1, mat1, 0);
        Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\GrayMap.jpg", mat1);

        //阈值处理,作用是为了更好的去做轮廓检测
        Imgproc.threshold(mat1, mat1, 75, 255, Imgproc.THRESH_BINARY_INV);//thresh为阈值
        //Imgproc.threshold(mat1, mat1, 75, 255, Imgproc.THRESH_BINARY);
        List<MatOfPoint> matOfPoints = new List<MatOfPoint>();

        Mat hierarchy = new Mat();

        //输出二值化图像
        Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\BinaryMap.jpg", mat1);

        //matOfPoints取得轮廓数;
        //Imgproc.findContours(mat1, matOfPoints, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        Imgproc.findContours(mat1, matOfPoints, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        //过滤不想要的轮廓
        List<MatOfPoint> filteredContours = new List<MatOfPoint>();
        foreach (MatOfPoint contour in matOfPoints)
        {
            double area = Imgproc.contourArea(contour);
            if (area >= 200)
            {
                filteredContours.Add(contour);
            }
        }
        Mat Apply = new Mat(0,0,CvType.CV_8UC4);
        mat.copyTo(Apply);


        //输出轮廓图
        double[] white = { 255f, 255f, 255f, 255f };
        Mat matEmpty = new Mat(400 * h / w, 400, CvType.CV_8UC4);
        for (int i = 0; i < matEmpty.rows(); i++)
        {
            for (int j = 0; j < matEmpty.cols(); j++)
            {
                matEmpty.put(i, j, white);
            }
        }
        /*string EmptyPaper = Application.dataPath + "/Resources/empty.jpg";
        Mat matEmpty = Imgcodecs.imread(EmptyPaper, 1);
        int we = matEmpty.cols();
        int he = matEmpty.rows();
        Imgproc.resize(matEmpty, matEmpty, new Size(400, 400 * he / we));
        Imgproc.GaussianBlur(matEmpty, matEmpty, new Size(400, 400 * he / we), 0);*/
        Imgproc.drawContours(matEmpty, filteredContours, -1, new Scalar(99, 0, 0), 3);
        Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\Contours.jpg", matEmpty);

        //用mat2表示绘制在texture上的轮廓，mat1作用在构建mesh上
        Mat mat2 = new Mat();
        mat1.copyTo(mat2);
        Core.flip(mat2, mat2, 0);
        Mat hier = new Mat();
        List<MatOfPoint> matOfPoints2 = new List<MatOfPoint>();
        Imgproc.findContours(mat2, matOfPoints2, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        List<MatOfPoint> filteredContours2 = new List<MatOfPoint>();
        foreach (MatOfPoint contour in matOfPoints2)
        {
            double area = Imgproc.contourArea(contour);
            if (area >200)
            {
                filteredContours2.Add(contour);
            }
        }
        Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\mat2.jpg", mat2);

        //表示轮廓的辅助矩阵
        //double[] white = { 255f, 255f, 255f, 255f };
        Mat matEmpty2 = new Mat(400 * h / w, 400, CvType.CV_8UC4);
        for (int i = 0; i < matEmpty2.rows(); i++)
        {
            for (int j = 0; j < matEmpty2.cols(); j++)
            {
                matEmpty2.put(i, j, white);
            }
        }
        Imgproc.drawContours(matEmpty2, filteredContours2, -1, new Scalar(0, 0, 0), 3);
        Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\Contours2.jpg", matEmpty2);

        //逐像素判断轮廓
        double[] data = { 0f, 0f, 0f,0f};//四通道
        double[] temp = new double[3] ;
        for (int i = 0; i < matEmpty2.rows(); i++)
        {
            for (int j = 0; j < matEmpty2.cols(); j++)
            {
                temp[0] = matEmpty2.get(i, j)[0];
                temp[1] = matEmpty2.get(i, j)[1];
                temp[2] = matEmpty2.get(i, j)[2];
                if (temp[0] == 0 && temp[1] == 0 && temp[2] == 0)
                {
                    Apply.put(i, j, data);
                    //Core.subtract(Apply, matEmpty2, Apply);
                }
                /*temp[0] = filteredContours2[0].get(i, j)[0];
                temp[1] = filteredContours2[0].get(i, j)[1];
                temp[2] = filteredContours2[0].get(i, j)[2];
                if (temp[0] != 255 && temp[1] != 255 && temp[2] != 255)
                {
                    Apply.put(i, j, data);
                    //Core.subtract(Apply, matEmpty2, Apply);
                }*/
            }
        }
        Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\Apply.png", Apply);
        Mat kernel = new Mat(3, 3, CvType.CV_32F);
        Imgproc.dilate(Apply,Apply, kernel);

        //轮廓图
        texture = new Texture2D(mat.cols(), mat.rows(), TextureFormat.BGRA32, false);
        texture = ConvertMatToTexture(Apply);
        //Utils.matToTexture2D(Apply, texture, true);
        //texture = ConvertMatToTexture(mat);

        texture1 = new Texture2D(mat.cols(), mat.rows(), TextureFormat.BGRA32, false);
        Utils.matToTexture2D(Apply, texture1, true);
        //texture1 = ConvertMatToTexture(Apply);

        Pic.texture = texture1;
        Pic.SetNativeSize();

        //存储轮廓
        List<MatOfPoint2f> newContours = new List<MatOfPoint2f>();
        foreach (MatOfPoint point in filteredContours)
        {
            MatOfPoint2f newPoint = new MatOfPoint2f(point.toArray());
            newContours.Add(newPoint);
        }
        MatOfPoint2f A = newContours[0];
        getApproximatedPoints(A);

    }

    void getApproximatedPoints(MatOfPoint2f contour)
    {
        List<Vector2> points = new List<Vector2>();
        foreach (OpenCVForUnity.CoreModule.Point p in contour.toList())
        {
            points.Add(new Vector2((float)p.x, (float)p.y));
        }

        // 创建x和y坐标的AnimationCurve
        AnimationCurve curveX = new AnimationCurve();
        AnimationCurve curveY = new AnimationCurve();

        // 为x和y坐标添加关键帧
        for (int i = 0; i < points.Count; i++)
        {
            curveX.AddKey((float)i / (points.Count - 1), points[i].x);
            curveY.AddKey((float)i / (points.Count - 1), points[i].y);
        }
        int numberOfPoints = 300; // 设置要生成的点数
        for (int i = 0; i < numberOfPoints; i++)
        {
            float t = (float)i / (numberOfPoints - 1);
            Vector3 point = new Vector3(curveX.Evaluate(t), curveY.Evaluate(t), 0);
            bounpoints.Add(point);
        }

    }
    void makeSubMesh(string path)
    {
        for (int i = 0; i < bounpoints.Count; i++)
        {
            polygon.Add(new Vertex(bounpoints[i].x, bounpoints[i].y));

            if (i == bounpoints.Count - 1)
            {
                polygon.Add(new Segment(new Vertex(bounpoints[i].x, bounpoints[i].y), new Vertex(bounpoints[0].x, bounpoints[0].y)));
            }
            else
            {
                polygon.Add(new Segment(new Vertex(bounpoints[i].x, bounpoints[i].y), new Vertex(bounpoints[i + 1].x, bounpoints[i + 1].y)));
            }
        }

        // 三角剖分


        var options = new ConstraintOptions() { ConformingDelaunay = true };
        var quality = new QualityOptions() { MinimumAngle = 34.2 };
        var mesh = polygon.Triangulate(options, quality);
        // 对内部进行采样


        List<int> outIndices = new List<int>();
        List<Vector3> outVertices = new List<Vector3>();
        foreach (ITriangle t in mesh.Triangles)
        {
            for (int j = 2; j >= 0; j--)
            {
                bool found = false;
                // 去重
                for (int k = 0; k < outVertices.Count; k++)
                {
                    if ((outVertices[k].x == t.GetVertex(j).X) && (outVertices[k].y == t.GetVertex(j).Y))
                    {
                        outIndices.Add(k);
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    outVertices.Add(new Vector3((float)t.GetVertex(j).X, (float)t.GetVertex(j).Y, 0));
                    outIndices.Add(outVertices.Count - 1);
                }
            }
        }

        //生成正面mesh
        Mesh mesh1 = new Mesh();
        mesh1.Clear();
        mesh1.vertices = outVertices.ToArray();
        mesh1.triangles = outIndices.ToArray();
        mesh1 = Subdivide(mesh1);


        // 网格几何体和顶点在内部重新排序，提高渲染性能
        mesh1.Optimize();
        mesh1.RecalculateNormals();

        //初始化反面mesh
        Mesh mesh2 = new Mesh
        {
            vertices = mesh1.vertices,
            triangles = mesh1.triangles,
            uv = mesh1.uv,
            normals = mesh1.normals,
            tangents = mesh1.tangents
        };


        //膨胀
        mesh1 = inflation(path, mesh1, -1);
        mesh2 = inflation(path, mesh2, 1);


        //为了得到反面mesh,让反面mesh法线朝外
        //创建新的三角形索引数组，用于存储反面的三角形索引
        int[] triangles = mesh2.triangles;
        int[] reverseTriangles = new int[triangles.Length];
        for (int i = 0; i < triangles.Length; i += 3)
        {
            // 将三角形的三个顶点索引按相反的顺序存储
            reverseTriangles[i] = triangles[i + 2];
            reverseTriangles[i + 1] = triangles[i + 1];
            reverseTriangles[i + 2] = triangles[i];
        }

        // 将反面的三角形索引数组赋值给网格的三角形索引数组
        mesh2.triangles = reverseTriangles;



        // Recalculate the normals and bounds of the mesh
        mesh1.RecalculateNormals();
        mesh1.RecalculateBounds();
        mesh2.RecalculateNormals();
        mesh2.RecalculateBounds();

        //赋值
        frontFace.GetComponent<MeshFilter>().sharedMesh = mesh1;
        backFace.GetComponent<MeshFilter>().sharedMesh = mesh2;

        //贴材质
        Material material = new Material(Shader.Find("Standard")); // 创建一个新材质，Shader为Standard                                                    //
        /*material.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
        material.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        material.SetInt("_ZWrite", 0);*/
        material.EnableKeyword("_ALPHAPREMULTIPLY_ON");
        material.SetColor("_Color", Color.white);
        material.mainTexture = texture; // 将导入的贴图赋值给材质的Main Texture属性
        MeshRenderer meshRenderer1 = frontFace.GetComponent<MeshRenderer>();
        MeshRenderer meshRenderer2 = backFace.GetComponent<MeshRenderer>();
        meshRenderer1.sharedMaterial = material;
        meshRenderer2.sharedMaterial = material;
        //SaveMaterial(material);

        //为了贴图建立UV
        Vector2[] uvs = new Vector2[mesh1.vertices.Length];
        for (int i = 0; i < uvs.Length; i++)
        {
             uvs[i] = new Vector2(mesh1.vertices[i].x / texture.width, mesh1.vertices[i].y / texture.height);

        }
        mesh1.uv = uvs;

        Vector2[] uvs2 = new Vector2[mesh2.vertices.Length];
        for (int i = 0; i < uvs.Length; i++)
        {
             uvs2[i] = new Vector2(mesh2.vertices[i].x / texture.width, mesh2.vertices[i].y / texture.height);

        }
        mesh2.uv = uvs2;
    }

    //将两个mesh组合
    void makeFinalMesh()
    {
        MeshFilter[] meshFilters = GetComponentsInChildren<MeshFilter>();
        CombineInstance[] combine = new CombineInstance[meshFilters.Length];
        Material[] mats = new Material[meshFilters.Length];
        Matrix4x4 matrix = transform.worldToLocalMatrix;
        for (int i = 0; i < meshFilters.Length; i++)
        {
            MeshFilter mf = meshFilters[i];
            MeshRenderer mr = meshFilters[i].GetComponent<MeshRenderer>();
            if (mr == null)
            {
                continue;
            }
            combine[i].mesh = mf.sharedMesh;
            combine[i].transform = matrix * mf.transform.localToWorldMatrix;
            mr.enabled = false;
            mats[i] = mr.sharedMaterial;
        }
        MeshFilter thisMeshFilter = GetComponent<MeshFilter>();
        meshF = new Mesh();
        thisMeshFilter.sharedMesh = meshF;
        meshF.CombineMeshes(combine, true);
        meshF = MeshScaler.ScaleMesh(meshF, new Vector3(0.1f, 0.1f, 0.1f));
        thisMeshFilter.sharedMesh = meshF;
        Vector3 center = meshF.bounds.center;
        Vector3[] vertices = meshF.vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] -= center;
        }
        meshF.vertices = vertices;
        meshF.RecalculateNormals();
        meshF.RecalculateBounds();
        MeshRenderer thisMeshRenderer = GetComponent<MeshRenderer>();
        thisMeshRenderer.sharedMaterials = mats;
        thisMeshRenderer.enabled = true;
        meshF.Optimize();

        //输出obj
        using (StreamWriter streamWriter = new StreamWriter(string.Format("{0}{1}.obj", Application.dataPath + "/Export/", this.meshF.name)))
        {
            streamWriter.Write(MeshToString(thisMeshFilter, new Vector3(-1f, 1f, 1f)));
            streamWriter.Close();
        }
        //AssetDatabase.Refresh();
    }

    //膨胀
    UnityEngine.Mesh inflation(string path, UnityEngine.Mesh mesh, int face)
    {
        //读入图像
        Mat mat = Imgcodecs.imread(path, 1);
        //mat = EnhancePic(mat);
        // 获得深度图
        Mat mat1 = pic2dept(mat);
        int w1 = mat1.cols();
        int h1 = mat1.rows();

        //获得深度图的texTure
        mat1.convertTo(mat1, CvType.CV_8UC1);


        Texture2D depthMap = new Texture2D(mat1.cols(), mat1.rows(), TextureFormat.BGRA32, false);
        Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\DepthMap.jpg", mat1);

        Utils.matToTexture2D(mat1, depthMap);

        Vector3[] vertices = mesh.vertices;

        //高度膨胀系数
        List<float> H = new List<float>();
        for (int i = 0; i < vertices.Length; i++)
        {
            // 获取2DMesh中的顶点
            Vector3 vertex = vertices[i];
            Vector2 position = new Vector2(vertex.x, vertex.y);


            // H初始化为灰度
            Color pixelColor = depthMap.GetPixel((int)position.x, (int)position.y);
            var c = (pixelColor.r + pixelColor.g + pixelColor.b) / 3f;
            float h = c * 60;//100 originally

            H.Add(h);
        }

        //平滑H
        float dMax = H.Max();
        for (int i = 0; i < H.Count(); ++i)
        {
            var a = dMax - H[i];
            H[i] = (float)Math.Sqrt(dMax * dMax - a * a);
        }
        float d = H.Max();

        //更新顶点
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 vertex = vertices[i];
            if (bounpoints.Contains(vertex))
            {
                Debug.Log("true");
                vertex.z = 0;
                continue;
            }

            vertex.z += (float)face * H[i];
            vertices[i] = vertex;
        }
        mesh.vertices = vertices;
        return mesh;
    }

    // //图像转为灰度图
    // Mat pic2dept(Mat inputMat)
    // {
    //     Mat outputMat;
    //     int w = inputMat.cols();
    //     int h = inputMat.rows();
    //     Imgproc.resize(inputMat, inputMat, new Size(400, 400 * h / w));
    //     Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\inputMap.jpg", inputMat);
    //     Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_BGR2GRAY);
    //     inputMat = ~inputMat;
    //     Imgproc.threshold(inputMat, inputMat, 20, 300, Imgproc.THRESH_BINARY);
    //     outputMat = new Mat(new Size(h, w), CvType.CV_32FC1);
    //     Imgproc.distanceTransform(inputMat, outputMat, Imgproc.DIST_L2, 3);
    //     // 标准化
    //     Core.normalize(outputMat, outputMat, 0, 255, Core.NORM_MINMAX);
    //     Imgcodecs.imwrite("D:\\Unity\\MeshPhoto\\GrayscaleMap.jpg", outputMat);
    //     Imgproc.resize(outputMat, outputMat, new Size(400, 400 * h / w));
    //     return (outputMat);

    // }

    Mat pic2dept(Mat inputMat)
{
    // 第一步：对输入图像进行预处理（如大小调整）
    int w = inputMat.Cols;
    int h = inputMat.Rows;
    Cv2.Resize(inputMat, inputMat, new Size(400, 400 * h / w));

    // 将图像转换为深度学习模型可以接受的格式
    Mat preprocessedImage = PrepareForModel(inputMat);

    // 加载预训练模型
    var graph = new TFGraph();
    var model = File.ReadAllBytes("path_to_model.pb");
    graph.Import(model);

    using (var session = new TFSession(graph))
    {
        // 创建输入张量
        var tensor = CreateTensorFromImage(preprocessedImage);
        var runner = session.GetRunner();
        runner.AddInput(graph["input"][0], tensor).Fetch(graph["output"][0]);
        
        // 执行模型推理
        var output = runner.Run();
        var outputTensor = output[0];

        // 将输出张量转换为图像
        Mat depthMap = ConvertTensorToMat(outputTensor, h, w);

        // 可选：保存深度图
        Cv2.ImWrite("D:\\Unity\\MeshPhoto\\DepthMap.jpg", depthMap);

        // 返回深度图
        return depthMap;
    }
}

Mat PrepareForModel(Mat inputMat)
{
    // 对图像进行归一化、通道变换等操作
    Cv2.CvtColor(inputMat, inputMat, ColorConversionCodes.BGR2RGB);
    inputMat.ConvertTo(inputMat, MatType.CV_32FC3, 1.0 / 255);
    return inputMat;
}

TFTensor CreateTensorFromImage(Mat image)
{
    // 将图像转换为 TensorFlow 使用的张量
    var shape = new TFShape(1, image.Rows, image.Cols, 3);
    return TFTensor.FromBuffer(shape, image.Data, 0, image.Rows * image.Cols * 3);
}

Mat ConvertTensorToMat(TFTensor tensor, int height, int width)
{
    // 将 TensorFlow 的输出张量转换为 OpenCV Mat
    var data = tensor.Data<float>();
    Mat mat = new Mat(height, width, MatType.CV_32FC1, data);
    Core.Normalize(mat, mat, 0, 255, NormTypes.MinMax);
    mat.ConvertTo(mat, MatType.CV_8UC1);
    return mat;
}

    //转化mat为texture
    public Texture2D ConvertMatToTexture(Mat mat)
    {
        int width = mat.width();
        int height = mat.height();

        Texture2D texture = new Texture2D(width, height, TextureFormat.RGBA32, false);

        // 将Mat应用到Texture2D
        Utils.matToTexture2D(mat, texture);

        // 在Texture2D上交换红蓝通道
        SwapRedAndBlueChannels(texture);

        return texture;
    }

    private void SwapRedAndBlueChannels(Texture2D texture)
    {
        Color[] pixels = texture.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            Color tempColor = pixels[i];
            pixels[i] = new Color(tempColor.b, tempColor.g, tempColor.r, tempColor.a);
        }

        texture.SetPixels(pixels);
        texture.Apply();
    }




    //三角形细分
    Mesh PerformLoopSubdivision(Mesh mesh)
    {
        Mesh newMesh = new Mesh();

        List<Vector3> newVertices = new List<Vector3>();
        List<int> newTriangles = new List<int>();

        // 处理现有顶点
        foreach (Vector3 vertex in mesh.vertices)
        {
            newVertices.Add(vertex);
        }

        // 处理现有三角形
        int[] triangles = mesh.triangles;
        for (int i = 0; i < triangles.Length; i += 3)
        {
            int a = triangles[i];
            int b = triangles[i + 1];
            int c = triangles[i + 2];

            // 计算边的中点
            Vector3 ab = (mesh.vertices[a] + mesh.vertices[b]) / 2;
            Vector3 bc = (mesh.vertices[b] + mesh.vertices[c]) / 2;
            Vector3 ca = (mesh.vertices[c] + mesh.vertices[a]) / 2;

            // 添加新顶点
            int abIndex = newVertices.Count;
            newVertices.Add(ab);

            int bcIndex = newVertices.Count;
            newVertices.Add(bc);

            int caIndex = newVertices.Count;
            newVertices.Add(ca);

            // 添加新三角形
            newTriangles.AddRange(new int[] { a, abIndex, caIndex });
            newTriangles.AddRange(new int[] { b, bcIndex, abIndex });
            newTriangles.AddRange(new int[] { c, caIndex, bcIndex });
            newTriangles.AddRange(new int[] { abIndex, bcIndex, caIndex });
        }

        newMesh.vertices = newVertices.ToArray();
        newMesh.triangles = newTriangles.ToArray();
        newMesh.RecalculateNormals();

        return newMesh;
    }

    private Mesh Subdivide(Mesh mesh)
    {
        Dictionary<Edge, int> newVertices = new Dictionary<Edge, int>();
        List<Vector3> vertices = new List<Vector3>(mesh.vertices);
        List<int> triangles = new List<int>(mesh.triangles);

        List<int> newTriangles = new List<int>();

        for (int i = 0; i < triangles.Count; i += 3)
        {
            int a = triangles[i];
            int b = triangles[i + 1];
            int c = triangles[i + 2];

            int ab = GetMidpointIndex(a, b, vertices, newVertices);
            int bc = GetMidpointIndex(b, c, vertices, newVertices);
            int ca = GetMidpointIndex(c, a, vertices, newVertices);

            newTriangles.AddRange(new int[] { a, ab, ca });
            newTriangles.AddRange(new int[] { b, bc, ab });
            newTriangles.AddRange(new int[] { c, ca, bc });
            newTriangles.AddRange(new int[] { ab, bc, ca });
        }

        Mesh subdividedMesh = new Mesh
        {
            vertices = vertices.ToArray(),
            triangles = newTriangles.ToArray()
        };

        subdividedMesh.RecalculateNormals();
        return subdividedMesh;
    }

    private int GetMidpointIndex(int a, int b, List<Vector3> vertices, Dictionary<Edge, int> newVertices)
    {
        Edge edge = new Edge(a, b);
        int index;

        if (newVertices.TryGetValue(edge, out index))
        {
            return index;
        }

        Vector3 newVertex = (vertices[a] + vertices[b]) * 0.5f;
        vertices.Add(newVertex);
        index = vertices.Count - 1;
        newVertices[edge] = index;

        if (IsBoundaryEdge(a, b, vertices))
        {
            bounpoints.Add(newVertex);
        }

        return index;
    }

    private bool IsBoundaryEdge(int a, int b, List<Vector3> vertices)
    {
        Vector3 vertexA = vertices[a];
        Vector3 vertexB = vertices[b];

        return bounpoints.Contains(vertexA) && bounpoints.Contains(vertexB);
    }

    public struct Edge
    {
        public int A { get; }
        public int B { get; }

        public Edge(int a, int b)
        {
            A = Mathf.Min(a, b);
            B = Mathf.Max(a, b);
        }

        public override bool Equals(object obj)
        {
            if (obj is Edge other)
            {
                return A == other.A && B == other.B;
            }

            return false;
        }

        public override int GetHashCode()
        {
            int hash = 17;
            hash = hash * 23 + A;
            hash = hash * 23 + B;
            return hash;
        }
    }


    public static class MeshScaler
    {
        public static Mesh ScaleMesh(Mesh mesh, Vector3 scale)
        {
            Mesh scaledMesh = new Mesh();
            scaledMesh.name = mesh.name + "_scaled";
            Vector3[] vertices = new Vector3[mesh.vertexCount];

            for (int i = 0; i < mesh.vertexCount; i++)
            {
                vertices[i] = Vector3.Scale(mesh.vertices[i], scale);
            }

            scaledMesh.vertices = vertices;
            scaledMesh.triangles = mesh.triangles;
            scaledMesh.normals = mesh.normals;
            scaledMesh.uv = mesh.uv;

            return scaledMesh;
        }
    }


    void transPosition()
    {
        this.transform.Rotate(-90f, 0, 0);
    }


    //public GameObject go;

    // Update is called once per frame
    void Update()
    {
    }


    //输出obj
    private string MeshToString(MeshFilter mf, Vector3 scale)
    {
        Mesh mesh = mf.mesh;
        Material[] sharedMaterials = mf.GetComponent<Renderer>().sharedMaterials;
        Vector2 textureOffset = mf.GetComponent<Renderer>().material.GetTextureOffset("_MainTex");
        Vector2 textureScale = mf.GetComponent<Renderer>().material.GetTextureScale("_MainTex");

        StringBuilder stringBuilder = new StringBuilder().Append("mtllib design.mtl")
            .Append("\n")
            .Append("g ")
            .Append(mf.name)
            .Append("\n");

        Vector3[] vertices = mesh.vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            Vector3 vector = vertices[i];
            stringBuilder.Append(string.Format("v {0} {1} {2}\n", vector.x * scale.x, vector.y * scale.y, vector.z * scale.z));
        }

        stringBuilder.Append("\n");

        Dictionary<int, int> dictionary = new Dictionary<int, int>();

        if (mesh.subMeshCount > 1)
        {
            int[] triangles = mesh.GetTriangles(1);

            for (int j = 0; j < triangles.Length; j += 3)
            {
                if (!dictionary.ContainsKey(triangles[j]))
                {
                    dictionary.Add(triangles[j], 1);
                }

                if (!dictionary.ContainsKey(triangles[j + 1]))
                {
                    dictionary.Add(triangles[j + 1], 1);
                }

                if (!dictionary.ContainsKey(triangles[j + 2]))
                {
                    dictionary.Add(triangles[j + 2], 1);
                }
            }
        }

        for (int num = 0; num != mesh.uv.Length; num++)
        {
            Vector2 vector2 = Vector2.Scale(mesh.uv[num], textureScale) + textureOffset;

            if (dictionary.ContainsKey(num))
            {
                stringBuilder.Append(string.Format("vt {0} {1}\n", mesh.uv[num].x, mesh.uv[num].y));
            }
            else
            {
                stringBuilder.Append(string.Format("vt {0} {1}\n", vector2.x, vector2.y));
            }
        }

        for (int k = 0; k < mesh.subMeshCount; k++)
        {
            stringBuilder.Append("\n");

            if (k == 0)
            {
                stringBuilder.Append("usemtl ").Append("Material_design").Append("\n");
            }

            if (k == 1)
            {
                stringBuilder.Append("usemtl ").Append("Material_logo").Append("\n");
            }

            int[] triangles2 = mesh.GetTriangles(k);

            for (int l = 0; l < triangles2.Length; l += 3)
            {
                stringBuilder.Append(string.Format("f {0}/{0} {1}/{1} {2}/{2}\n", triangles2[l] + 1, triangles2[l + 2] + 1, triangles2[l + 1] + 1));
            }
        }

        return stringBuilder.ToString();
    }
}
