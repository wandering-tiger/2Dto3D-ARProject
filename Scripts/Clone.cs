using System;
using System.Collections;
using System.Collections.Generic;
using Lean.Common;
using UnityEngine;
using UnityEngine.UI;

public class Clone : MonoBehaviour
{
    public GameObject dog;
    public GameObject elephant;
    public GameObject cloud;
    List<GameObject> objs = new List<GameObject>();
    // Start is called before the first frame update
    void Start()
    {
        dog = GameObject.Find("ImageTarget1/StillDog");
        elephant = GameObject.Find("ImageTarget2/Elephant");
        cloud = GameObject.Find("ImageTarget3/Cloud");
        objs.Add(dog);
        objs.Add (elephant);
        objs.Add(cloud);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void Click_t()
    {
        foreach (GameObject obj in objs)
        {
            if(obj.GetComponent<LeanSelectable>().IsSelected)
            {
                try
                {
                    GameObject clone = Instantiate(obj, obj.transform.position, obj.transform.rotation);
                }
                catch (NullReferenceException e)
                {
                    Console.WriteLine("Exception caught: ", e);
                }
                break;
            }
        }
    }
}
