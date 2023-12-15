using System.Collections;
using System.Collections.Generic;
using Lean.Touch;
using UnityEngine;

public class Button2 : MonoBehaviour
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
        dog.GetComponent<LeanPinchScale>().enabled = true;
        dog.GetComponent<LeanTwistRotate>().enabled = false;
        elephant.GetComponent<LeanDragTranslate>().enabled = false;
        elephant.GetComponent<LeanPinchScale>().enabled = true;
        elephant.GetComponent<LeanTwistRotate>().enabled = false;
        cloud.GetComponent<LeanDragTranslate>().enabled = false;
        cloud.GetComponent<LeanPinchScale>().enabled = true;
        cloud.GetComponent<LeanTwistRotate>().enabled = false;
        Debug.Log("Scale!");
    }
}
