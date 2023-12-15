using System.Collections;
using System.Collections.Generic;
using Lean.Touch;
using UnityEngine;

public class Button3 : MonoBehaviour
{
    //add all the objects here
    public GameObject dog;
    public GameObject elephant;
    public GameObject cloud;
    // Start is called before the first frame update
    void Start()
    {
        dog = GameObject.Find("ImageTarget1/StillDog");
        elephant = GameObject.Find("ImageTarget2/Elephant");
        cloud = GameObject.Find("ImageTarget3/Cloud");
    }

    // Update is called once per frame
    void Update()
    {

    }
    public void Click_test()
    {
        dog.GetComponent<LeanDragTranslate>().enabled = false;
        dog.GetComponent<LeanPinchScale>().enabled = false;
        dog.GetComponent<LeanTwistRotate>().enabled = true;
        elephant.GetComponent<LeanDragTranslate>().enabled = false;
        elephant.GetComponent<LeanPinchScale>().enabled = false;
        elephant.GetComponent<LeanTwistRotate>().enabled = true;
        cloud.GetComponent<LeanDragTranslate>().enabled = false;
        cloud.GetComponent<LeanPinchScale>().enabled = false;
        cloud.GetComponent<LeanTwistRotate>().enabled = true;
        Debug.Log("Rotate!");
    }
}
