using System;
using System.Collections;
using System.Collections.Generic;
using Lean.Touch;
using UnityEngine;

public class Button1 : MonoBehaviour
{
    //add all the objects here
    public GameObject dog;
    public GameObject elephant;
    public GameObject cloud;
    public GameObject cloud_clone;
    // Start is called before the first frame update
    void Start()
    {
        dog = GameObject.Find("ImageTarget1/StillDog");
        elephant = GameObject.Find("ImageTarget2/Elephant");
        cloud =  GameObject.Find("ImageTarget3/Cloud");
        //This is useless because Cloud(Clone) is always the same as its father.
        try
        {
            cloud_clone = GameObject.Find("Cloud(Clone)");
        }
        catch (NullReferenceException e)
        {
            Console.WriteLine("Exception caught: ", e);
        }  
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public void Click_test()
    {
        dog.GetComponent<LeanDragTranslate>().enabled = true;
        dog.GetComponent<LeanPinchScale>().enabled = false;
        dog.GetComponent<LeanTwistRotate>().enabled = false;
        elephant.GetComponent<LeanDragTranslate>().enabled = true;
        elephant.GetComponent<LeanPinchScale>().enabled = false;
        elephant.GetComponent<LeanTwistRotate>().enabled = false;
        cloud.GetComponent<LeanDragTranslate>().enabled = true;
        cloud.GetComponent<LeanPinchScale>().enabled = false;
        cloud.GetComponent<LeanTwistRotate>().enabled = false;
        //This is useless.
        try
        {
            cloud_clone.GetComponent<LeanDragTranslate>().enabled = true;
        }
        catch (NullReferenceException e)
        {
            Console.WriteLine("Exception caught: ", e);
        }
        Debug.Log("Translate!");
    }
}
