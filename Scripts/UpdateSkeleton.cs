using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UpdateSkeleton : MonoBehaviour
{
    public string[] joints; 
    public class Joint
    {
        public string name; //关节名字
        public Vector3 P; //关节自身位置
        public Vector3 Jstart; // 起始关节的位置
        public Vector3 Jend; // 结束关节的位置
        public float r1; // 起始关节到计算节点 J 的长度比率
        public float r2; // 结束关节到计算节点 J 的长度比率
        public Vector3 D; // 从结束关节指向起始关节的偏移向量
        public int F;
        public int S;
        public Joint(string name,Vector3 P,float r1, float r2, Vector3 D,int F, int S)
        {
            this.name = name;
            this.P = P;
            this.r1 = r1;
            this.r2 = r2;
            this.D = D;
            this.F = F;
            this.S = S;
            //this.Jstart = Jstart;
            //this.Jend = Jend;
        }
        public void setP(Vector3 P)
        {
            this.P = P;
        }
        public Vector3 getP()
        {
            return this.P;
        }
    }
    public Vector3 nanForV;
    public List<Joint> J;

    private void Start()
    {
        nanForV = new Vector3(float.NaN, float.NaN, float.NaN);
        J = new List<Joint>();
        joints = new string[17];
        InitJoints();
        for (int i = 0; i < J.Count; ++i)
        {
            joints[i] = J[i].name;
        }
    }


    public void InitJoints()
    {
        //Spine、Hips、Neck、Head、LeftShoulder、LeftArm、LeftHand、RightShoulder、RightArm、RightHand、LeftUpperLeg、LeftLowerLeg、LeftFoot、RightUpperLeg、RightLowerLeg、RightFoot
        //0
        J.Add(new Joint("Spine", new Vector3(0, 0, 0), 0.1f, 0.2f, new Vector3(0, 0, 0), 0, 0));
        //1
        J.Add(new Joint("Hips", nanForV, 0.5f, 0.5f, (new Vector3(0, 0, 0)).normalized, 10, 13));
        //2
        J.Add(new Joint("Neck", nanForV, 0.5f, 0.5f, new Vector3(0, 0, 0), 4, 7));
        //3
        J.Add(new Joint("Head", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 2, -1));
        //4
        J.Add(new Joint("LeftShoulder", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 2, 5));
        //5
        J.Add(new Joint("LeftArm", nanForV, 0.5f, 0.5f, new Vector3(0, 0, 0), 4, 6));
        //6
        J.Add(new Joint("LeftHand", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 5, -1));
        //7
        J.Add(new Joint("RightShoulder", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 2, 8));
        //8
        J.Add(new Joint("RightArm", nanForV, 0.5f, 0.5f, new Vector3(0, 0, 0), 7, 9));
        //9
        J.Add(new Joint("RightHand", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 8, -1));
        //10
        J.Add(new Joint("LeftUpperLeg", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 1, 11));
        //11
        J.Add(new Joint("LeftLowerLeg", nanForV, 0.4f, 0.6f, new Vector3(0, 0, 0), 10, 12));
        //12
        J.Add(new Joint("LeftFoot", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 11, -1));
        //13
        J.Add(new Joint("RightUpperLeg", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 1, 14));
        //14
        J.Add(new Joint("RightLowerLeg", nanForV, 0.5f, 0.5f, new Vector3(0, 0, 0), 13, 15));
        //15
        J.Add(new Joint("RightFoot", nanForV, 0.1f, 0.2f, new Vector3(0, 0, 0), 14, -1));

    }

    public Vector3 ComputeJointPosition(string name, Joint joint)
    {
        Vector3 Jstart = J[joint.F].P;
        Vector3 Jend = J[joint.S].P;
        // 如果起始关节和结束关节都已经定义，则可以直接计算节点位置
        if (Jstart != null && Jend != null)
        {
            joint.D = Vector3.Normalize(Jend - Jstart);
            Vector3 direction = Jend - Jstart;
            float distance = direction.magnitude;

            // 计算节点相对于 Jstart 和 Jend 的位置偏移量
            Vector3 offset = joint.r1 * direction;

            // 计算节点相对于 Jstart 和 Jend 的方向偏移量
            offset += joint.r2 * distance * joint.D;
            // 计算节点的绝对位置
            Vector3 position = Jstart + offset;
            return position;
        }
        // 如果起始关节和结束关节有一个或两个未定义，则需要递归调用 ComputeJointPosition 函数
        else
        {
            if (Jstart == nanForV)
            {
                Jstart = ComputeJointPosition(name, GetJointDependency(joint, true));
            }
            if (Jend == nanForV)
            {
                Jend = ComputeJointPosition(name, GetJointDependency(joint, false));
            }
            return ComputeJointPosition(name, joint);
        }
    }

    private Joint GetJointDependency(Joint joint, bool isStart)
    {
        // 根据需要获取起始关节或结束关节的位置
        Joint dependency = isStart ? J[joint.F] : J[joint.S];
        Debug.Log("recc");

        // 如果依赖关节已经定义，则直接返回
        if (dependency.P != nanForV)
        {
            return dependency;
        }
        // 如果依赖关节未定义，则递归调用 GetJointDependency 函数
        else
        {
            return GetJointDependency(joint, isStart);
        }
    }

    
    public void SetPosition(string name,Vector3 pos)
    {
        int i = Array.IndexOf(joints, name);
        J[i].P = pos;
        GameObject.FindWithTag(name).transform.position = pos;
        Debug.Log(name + pos);
    }

    public Vector3 GetNewSkeleton(string name)
    {
        int i = Array.IndexOf(joints, name);
        J[i].setP(ComputeJointPosition(name, J[i]));
        return J[i].getP();
    }
"
    public void refreshSkeleton()
    {
        Vector3 z1 = new Vector3(0, 0, -5);
        Vector3 z2 = new Vector3(0, 0, -10);
        Vector3 z3 = new Vector3(0, 0, -15);
        GameObject parent = GameObject.Find("Test");
        GameObject.Find("Test/Spine/Neck").GetComponent<Transform>().position = GetNewSkeleton("Neck")+z1;
        GameObject.Find("Test/Spine/Neck/LeftShoulder/LeftArm").GetComponent<Transform>().position = GetNewSkeleton("LeftArm")+z2;
        GameObject.Find("Test/Spine/Neck/RightShoulder/RightArm").GetComponent<Transform>().position = GetNewSkeleton("RightArm")+z2;
        GameObject.Find("Test/Spine/Hips").GetComponent<Transform>().position = GetNewSkeleton("Hips")+z1;
        GameObject.Find("Test/Spine/Hips/LeftUpperLeg/LeftLowerLeg").GetComponent<Transform>().position = GetNewSkeleton("LeftLowerLeg")+z2;
        GameObject.Find("Test/Spine/Hips/RightUpperLeg/RightLowerLeg").GetComponent<Transform>().position = GetNewSkeleton("RightLowerLeg")+z2;
        GameObject.FindWithTag("Head").GetComponent<Transform>().position += z1;
        GameObject.FindWithTag(LeftHand").GetComponent<Transform>().position += z3;
        GameObject.FindWithTag("RightHand").GetComponent<Transform>().position += z3;
        GameObject.FindWithTag("LeftUpperLeg").GetComponent<Transform>().position += z1;
        GameObject.FindWithTag("RightUpperLeg").GetComponent<Transform>().position += z1;
        GameObject.FindWithTag("LeftFoot").GetComponent<Transform>().position += z1;
        GameObject.FindWithTag("RightFoot").GetComponent<Transform>().position += z1;


    }
}

