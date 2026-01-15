using UnityEngine;
using Microsoft.MixedReality.OpenXR;


public class test : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject cubeObject;

    void Start()
    {
        // Cube 오브젝트에 SpatialAnchor 붙이기
        var anchor = gameObject.AddComponent<SpatialAnchor>();
        Debug.Log("SpatialAnchor attached to Cube.");
    }

    // Update is called once per frame
    void Update()
    {
        
    }

}
