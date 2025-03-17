uniform sampler2D base; 
uniform sampler2D normalMap; 
uniform vec3 CAMERA_POSITION; 
varying vec3 position; 
varying vec3 lightvec; 
 
void main() 
{   
  vec3 norm = texture2D(normalMap, gl_TexCoord[0].st).rgb * 2.0 - 1.0;   
  vec3 baseColor = texture2D(base, gl_TexCoord[0].st).rgb; 
   
  float dist = length(lightvec);   
   
  vec3 lightVector = normalize(lightvec);                 
  float nxDir = max(0.0, dot(norm, lightVector));  
  vec4 diffuse = gl_LightSource[0].diffuse * nxDir; 
   
  float specularPower = 0.0; 
  if(nxDir != 0.0) 
  { 
    vec3 cameraVector = normalize(CAMERA_POSITION - position.xyz);   
    vec3 halfVector = normalize(lightVector + cameraVector); 
    float nxHalf = max(0.0,dot(norm, halfVector));       
    specularPower = pow(nxHalf, gl_FrontMaterial.shininess); 
  } 
  vec4 specular = gl_LightSource[0].specular * specularPower; 
       
  gl_FragColor =  gl_LightSource[0].ambient +  
                 (diffuse * vec4(baseColor.rgb,1.0)) + 
                  specular;   
} 